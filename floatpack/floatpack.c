#include "floatpack.h"
#include "fp4_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

// ============================================================
// Vulkan state (singleton)
// ============================================================

static VkInstance             vk_instance        = VK_NULL_HANDLE;
static VkPhysicalDevice       vk_phys_device     = VK_NULL_HANDLE;
static VkDevice               vk_device          = VK_NULL_HANDLE;
static VkQueue                vk_queue           = VK_NULL_HANDLE;
static VkCommandPool          vk_cmd_pool        = VK_NULL_HANDLE;
static VkDescriptorPool       vk_desc_pool       = VK_NULL_HANDLE;
static VkDescriptorSetLayout  vk_desc_set_layout = VK_NULL_HANDLE;
static VkPipelineLayout       vk_pipeline_layout = VK_NULL_HANDLE;
static VkPipeline             vk_pipeline        = VK_NULL_HANDLE;
static VkBuffer               vk_lut_buffer      = VK_NULL_HANDLE;
static VkDeviceMemory         vk_lut_memory      = VK_NULL_HANDLE;
static VkShaderModule         vk_shader_module   = VK_NULL_HANDLE;
static VkShaderModule         vk_shader_module_q4= VK_NULL_HANDLE;
static VkPipeline             vk_pipeline_q4     = VK_NULL_HANDLE;
static VkDescriptorSet        vk_desc_set        = VK_NULL_HANDLE;

static int vk_initialized = 0;

// Forward declarations for cleanup helpers
static void cleanup_after_lut(void);
static void cleanup_after_pool(void);

// ============================================================
// SPIR-V loader (reads .spv file)
// ============================================================

static char* read_file(const char *path, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    if (sz <= 0) { fclose(fp); return NULL; }
    rewind(fp);
    char *data = (char*)malloc((size_t)sz);
    if (!data) { fclose(fp); return NULL; }
    fread(data, 1, (size_t)sz, fp);
    fclose(fp);
    *out_size = (size_t)sz;
    return data;
}

// ============================================================
// Initialization
// ============================================================

int floatpack_init(void) {
    if (vk_initialized) return 0;

    // ---- Instance ----
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "floatpack",
        .applicationVersion = 1,
        .pEngineName = "floatpack",
        .engineVersion = 1,
        .apiVersion = VK_API_VERSION_1_2,
    };
    VkInstanceCreateInfo inst_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = NULL,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = NULL,
    };
    if (vkCreateInstance(&inst_info, NULL, &vk_instance) != VK_SUCCESS) {
        fprintf(stderr, "floatpack: failed to create Vulkan instance\n");
        return -1;
    }

    // ---- Physical device ----
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(vk_instance, &dev_count, NULL);
    if (dev_count == 0) {
        fprintf(stderr, "floatpack: no Vulkan devices found\n");
        vkDestroyInstance(vk_instance, NULL);
        vk_instance = VK_NULL_HANDLE;
        return -1;
    }
    VkPhysicalDevice *devices = (VkPhysicalDevice*)malloc(dev_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(vk_instance, &dev_count, devices);
    // Pick first discrete GPU, else first device
    vk_phys_device = devices[0];
    for (uint32_t i = 0; i < dev_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            vk_phys_device = devices[i];
            break;
        }
        // prefer integrated GPU over CPU
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            vk_phys_device = devices[i];
        }
    }
    free(devices);

    // ---- Queue family ----
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk_phys_device, &queue_family_count, NULL);
    VkQueueFamilyProperties *qf_props = (VkQueueFamilyProperties*)malloc(
        queue_family_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(vk_phys_device, &queue_family_count, qf_props);
    uint32_t queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_family = i;
            break;
        }
    }
    free(qf_props);
    if (queue_family == UINT32_MAX) {
        fprintf(stderr, "floatpack: no compute queue family\n");
        vkDestroyInstance(vk_instance, NULL);
        vk_instance = VK_NULL_HANDLE;
        return -1;
    }

    // ---- Logical device ----
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };
    VkDeviceCreateInfo dev_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = NULL,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = NULL,
    };
    if (vkCreateDevice(vk_phys_device, &dev_info, NULL, &vk_device) != VK_SUCCESS) {
        fprintf(stderr, "floatpack: failed to create logical device\n");
        vkDestroyInstance(vk_instance, NULL);
        vk_instance = VK_NULL_HANDLE;
        return -1;
    }
    vkGetDeviceQueue(vk_device, queue_family, 0, &vk_queue);

    // ---- Command pool ----
    VkCommandPoolCreateInfo cmd_pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue_family,
    };
    if (vkCreateCommandPool(vk_device, &cmd_pool_info, NULL, &vk_cmd_pool) != VK_SUCCESS) {
        fprintf(stderr, "floatpack: failed to create command pool\n");
        vkDestroyDevice(vk_device, NULL);
        vkDestroyInstance(vk_instance, NULL);
        vk_device = VK_NULL_HANDLE;
        vk_instance = VK_NULL_HANDLE;
        return -1;
    }

    // ---- Descriptor pool ----
    VkDescriptorPoolSize pool_sizes[2] = {
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  .descriptorCount = 32 },
        { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  .descriptorCount = 32 },
    };
    VkDescriptorPoolCreateInfo desc_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 32,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes,
    };
    if (vkCreateDescriptorPool(vk_device, &desc_pool_info, NULL, &vk_desc_pool) != VK_SUCCESS) {
        fprintf(stderr, "floatpack: failed to create descriptor pool\n");
        vkDestroyCommandPool(vk_device, vk_cmd_pool, NULL);
        vkDestroyDevice(vk_device, NULL);
        vkDestroyInstance(vk_instance, NULL);
        vk_cmd_pool = VK_NULL_HANDLE;
        vk_device = VK_NULL_HANDLE;
        vk_instance = VK_NULL_HANDLE;
        return -1;
    }

    // ---- LUT buffer ----
    float lut_data[FP4_LUT_SIZE];
    fp4_build_lut(lut_data);

    VkBufferCreateInfo lut_buf_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = FP4_LUT_SIZE * sizeof(float),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(vk_device, &lut_buf_info, NULL, &vk_lut_buffer) != VK_SUCCESS) {
        fprintf(stderr, "floatpack: failed to create LUT buffer\n");
        cleanup_after_pool();
        return -1;
    }

    VkMemoryRequirements lut_mem_req;
    vkGetBufferMemoryRequirements(vk_device, vk_lut_buffer, &lut_mem_req);

    VkPhysicalDeviceMemoryProperties phys_mem_props;
    vkGetPhysicalDeviceMemoryProperties(vk_phys_device, &phys_mem_props);

    uint32_t mem_type_idx = UINT32_MAX;
    for (uint32_t i = 0; i < phys_mem_props.memoryTypeCount; i++) {
        if ((lut_mem_req.memoryTypeBits & (1u << i)) &&
            (phys_mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            mem_type_idx = i;
            break;
        }
    }
    if (mem_type_idx == UINT32_MAX) {
        fprintf(stderr, "floatpack: no suitable memory type for LUT\n");
        vkDestroyBuffer(vk_device, vk_lut_buffer, NULL);
        cleanup_after_pool();
        return -1;
    }

    VkMemoryAllocateInfo lut_alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = lut_mem_req.size,
        .memoryTypeIndex = mem_type_idx,
    };
    if (vkAllocateMemory(vk_device, &lut_alloc_info, NULL, &vk_lut_memory) != VK_SUCCESS) {
        fprintf(stderr, "floatpack: failed to allocate LUT memory\n");
        vkDestroyBuffer(vk_device, vk_lut_buffer, NULL);
        cleanup_after_pool();
        return -1;
    }
    vkBindBufferMemory(vk_device, vk_lut_buffer, vk_lut_memory, 0);

    void *lut_map;
    vkMapMemory(vk_device, vk_lut_memory, 0, VK_WHOLE_SIZE, 0, &lut_map);
    memcpy(lut_map, lut_data, FP4_LUT_SIZE * sizeof(float));
    vkUnmapMemory(vk_device, vk_lut_memory);

    // ---- Descriptor set layout ----
    VkDescriptorSetLayoutBinding bindings[4] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
    };
    VkDescriptorSetLayoutCreateInfo ds_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = bindings,
    };
    if (vkCreateDescriptorSetLayout(vk_device, &ds_layout_info, NULL, &vk_desc_set_layout) != VK_SUCCESS) {
        fprintf(stderr, "floatpack: failed to create descriptor set layout\n");
        cleanup_after_lut();
        return -1;
    }

    // ---- Pipeline layout ----
    VkPushConstantRange pc_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 3 * sizeof(int),
    };
    VkPipelineLayoutCreateInfo pl_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &vk_desc_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pc_range,
    };
    if (vkCreatePipelineLayout(vk_device, &pl_layout_info, NULL, &vk_pipeline_layout) != VK_SUCCESS) {
        fprintf(stderr, "floatpack: failed to create pipeline layout\n");
        vkDestroyDescriptorSetLayout(vk_device, vk_desc_set_layout, NULL);
        cleanup_after_lut();
        return -1;
    }

    // ---- Load SPIR-V shader from file ----
    size_t spv_size = 0;
    char *spv_data = read_file("floatpack_fp4.spv", &spv_size);
    if (!spv_data) {
        spv_data = read_file("../floatpack/floatpack_fp4.spv", &spv_size);
    }
    if (!spv_data) {
        spv_data = read_file("/home/alivetilleve/llama.cpp-floatpack/floatpack/floatpack_fp4.spv", &spv_size);
    }
    if (!spv_data) {
        fprintf(stderr, "floatpack: cannot load floatpack_fp4.spv\n");
        fprintf(stderr, "  Compile the shader with:\n");
        fprintf(stderr, "    glslc floatpack_fp4.comp -o floatpack_fp4.spv\n");
        vkDestroyPipelineLayout(vk_device, vk_pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(vk_device, vk_desc_set_layout, NULL);
        cleanup_after_lut();
        return -1;
    }

    VkShaderModuleCreateInfo shader_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_size,
        .pCode = (const uint32_t*)spv_data,
    };
    if (vkCreateShaderModule(vk_device, &shader_info, NULL, &vk_shader_module) != VK_SUCCESS) {
        fprintf(stderr, "floatpack: failed to create shader module\n");
        free(spv_data);
        vkDestroyPipelineLayout(vk_device, vk_pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(vk_device, vk_desc_set_layout, NULL);
        cleanup_after_lut();
        return -1;
    }
    free(spv_data);

    VkPipelineShaderStageCreateInfo stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = vk_shader_module,
        .pName = "main",
    };
    VkComputePipelineCreateInfo pipe_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage_info,
        .layout = vk_pipeline_layout,
    };
    if (vkCreateComputePipelines(vk_device, VK_NULL_HANDLE, 1, &pipe_info, NULL, &vk_pipeline) != VK_SUCCESS) {
        fprintf(stderr, "floatpack: failed to create compute pipeline\n");
        vkDestroyShaderModule(vk_device, vk_shader_module, NULL);
        vkDestroyPipelineLayout(vk_device, vk_pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(vk_device, vk_desc_set_layout, NULL);
        cleanup_after_lut();
        return -1;
    }

    // ---- Allocate descriptor set ----
    VkDescriptorSetAllocateInfo desc_set_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk_desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &vk_desc_set_layout,
    };
    if (vkAllocateDescriptorSets(vk_device, &desc_set_alloc, &vk_desc_set) != VK_SUCCESS) {
        fprintf(stderr, "floatpack: failed to allocate descriptor set\n");
        vkDestroyPipeline(vk_device, vk_pipeline, NULL);
        vkDestroyShaderModule(vk_device, vk_shader_module, NULL);
        vkDestroyPipelineLayout(vk_device, vk_pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(vk_device, vk_desc_set_layout, NULL);
        cleanup_after_lut();
        return -1;
    }

    // ---- Load Q4_0 fused shader ----
    size_t spv_q4_size = 0;
    char *spv_q4_data = read_file("floatpack_q4_0.spv", &spv_q4_size);
    if (!spv_q4_data) {
        spv_q4_data = read_file("../floatpack/floatpack_q4_0.spv", &spv_q4_size);
    }
    if (!spv_q4_data) {
        spv_q4_data = read_file("/home/alivetilleve/llama.cpp-floatpack/floatpack/floatpack_q4_0.spv", &spv_q4_size);
    }
    if (!spv_q4_data) {
        fprintf(stderr, "floatpack: cannot load floatpack_q4_0.spv\n");
        vkDestroyPipeline(vk_device, vk_pipeline, NULL);
        vkDestroyShaderModule(vk_device, vk_shader_module, NULL);
        vkDestroyPipelineLayout(vk_device, vk_pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(vk_device, vk_desc_set_layout, NULL);
        cleanup_after_lut();
        return -1;
    }

    VkShaderModuleCreateInfo shader_q4_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_q4_size,
        .pCode = (const uint32_t*)spv_q4_data,
    };
    if (vkCreateShaderModule(vk_device, &shader_q4_info, NULL, &vk_shader_module_q4) != VK_SUCCESS) {
        fprintf(stderr, "floatpack: failed to create Q4_0 shader module\n");
        free(spv_q4_data);
        vkDestroyPipeline(vk_device, vk_pipeline, NULL);
        vkDestroyShaderModule(vk_device, vk_shader_module, NULL);
        vkDestroyPipelineLayout(vk_device, vk_pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(vk_device, vk_desc_set_layout, NULL);
        cleanup_after_lut();
        return -1;
    }
    free(spv_q4_data);

    // Create Q4_0 pipeline (uses same layout as FP4 pipeline)
    VkPipelineShaderStageCreateInfo q4_stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = vk_shader_module_q4,
        .pName = "main",
    };
    VkComputePipelineCreateInfo q4_pipe_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = q4_stage_info,
        .layout = vk_pipeline_layout,
    };
    if (vkCreateComputePipelines(vk_device, VK_NULL_HANDLE, 1, &q4_pipe_info, NULL, &vk_pipeline_q4) != VK_SUCCESS) {
        fprintf(stderr, "floatpack: failed to create Q4_0 compute pipeline\n");
        vkDestroyShaderModule(vk_device, vk_shader_module_q4, NULL);
        vkDestroyPipeline(vk_device, vk_pipeline, NULL);
        vkDestroyShaderModule(vk_device, vk_shader_module, NULL);
        vkDestroyPipelineLayout(vk_device, vk_pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(vk_device, vk_desc_set_layout, NULL);
        cleanup_after_lut();
        return -1;
    }

    vk_initialized = 1;
    return 0;
}

// cleanup helpers
static void cleanup_after_lut(void) {
    vkFreeMemory(vk_device, vk_lut_memory, NULL);
    vkDestroyBuffer(vk_device, vk_lut_buffer, NULL);
    cleanup_after_pool();
}

static void cleanup_after_pool(void) {
    vkDestroyDescriptorPool(vk_device, vk_desc_pool, NULL);
    vkDestroyCommandPool(vk_device, vk_cmd_pool, NULL);
    vkDestroyDevice(vk_device, NULL);
    vkDestroyInstance(vk_instance, NULL);
    vk_cmd_pool = VK_NULL_HANDLE;
    vk_device = VK_NULL_HANDLE;
    vk_instance = VK_NULL_HANDLE;
}

void floatpack_cleanup(void) {
    if (!vk_initialized) return;
    vkDestroyPipeline(vk_device, vk_pipeline_q4, NULL);
    vkDestroyShaderModule(vk_device, vk_shader_module_q4, NULL);
    vkDestroyPipeline(vk_device, vk_pipeline, NULL);
    vkDestroyShaderModule(vk_device, vk_shader_module, NULL);
    vkDestroyPipelineLayout(vk_device, vk_pipeline_layout, NULL);
    vkDestroyDescriptorSetLayout(vk_device, vk_desc_set_layout, NULL);
    vkFreeMemory(vk_device, vk_lut_memory, NULL);
    vkDestroyBuffer(vk_device, vk_lut_buffer, NULL);
    vkDestroyDescriptorPool(vk_device, vk_desc_pool, NULL);
    vkDestroyCommandPool(vk_device, vk_cmd_pool, NULL);
    vkDestroyDevice(vk_device, NULL);
    vkDestroyInstance(vk_instance, NULL);
    vk_pipeline_q4 = VK_NULL_HANDLE;
    vk_shader_module_q4 = VK_NULL_HANDLE;
    vk_pipeline = VK_NULL_HANDLE;
    vk_shader_module = VK_NULL_HANDLE;
    vk_pipeline_layout = VK_NULL_HANDLE;
    vk_desc_set_layout = VK_NULL_HANDLE;
    vk_lut_memory = VK_NULL_HANDLE;
    vk_lut_buffer = VK_NULL_HANDLE;
    vk_desc_pool = VK_NULL_HANDLE;
    vk_cmd_pool = VK_NULL_HANDLE;
    vk_device = VK_NULL_HANDLE;
    vk_instance = VK_NULL_HANDLE;
    vk_initialized = 0;
}

int floatpack_ready(void) {
    return vk_initialized;
}

// ============================================================
// Per-call buffer allocation helper
// ============================================================

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
} fp_vk_buffer;

static int create_device_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags props, fp_vk_buffer *out) {
    VkBufferCreateInfo buf_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(vk_device, &buf_info, NULL, &out->buffer) != VK_SUCCESS) return -1;

    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(vk_device, out->buffer, &mem_req);

    VkPhysicalDeviceMemoryProperties phys_mem;
    vkGetPhysicalDeviceMemoryProperties(vk_phys_device, &phys_mem);

    uint32_t mem_type = UINT32_MAX;
    for (uint32_t i = 0; i < phys_mem.memoryTypeCount; i++) {
        if ((mem_req.memoryTypeBits & (1u << i)) &&
            (phys_mem.memoryTypes[i].propertyFlags & props) == props) {
            mem_type = i;
            break;
        }
    }
    if (mem_type == UINT32_MAX) {
        vkDestroyBuffer(vk_device, out->buffer, NULL);
        return -1;
    }

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_req.size,
        .memoryTypeIndex = mem_type,
    };
    if (vkAllocateMemory(vk_device, &alloc_info, NULL, &out->memory) != VK_SUCCESS) {
        vkDestroyBuffer(vk_device, out->buffer, NULL);
        return -1;
    }
    vkBindBufferMemory(vk_device, out->buffer, out->memory, 0);
    out->size = (size_t)size;
    return 0;
}

static void destroy_device_buffer(fp_vk_buffer *buf) {
    vkDestroyBuffer(vk_device, buf->buffer, NULL);
    vkFreeMemory(vk_device, buf->memory, NULL);
    buf->buffer = VK_NULL_HANDLE;
    buf->memory = VK_NULL_HANDLE;
    buf->size = 0;
}

// ============================================================
// Matmul
// ============================================================

int floatpack_matmul_fp4(int M, int N, int K_div4,
                         const uint16_t *A_host, const uint16_t *B_host, float *C_host) {
    if (!vk_initialized) return -1;

    size_t size_A = (size_t)M * K_div4 * sizeof(uint16_t);
    size_t size_B = (size_t)N * K_div4 * sizeof(uint16_t);
    size_t size_C = (size_t)M * N * sizeof(float);

    fp_vk_buffer buf_A = {0}, buf_B = {0}, buf_C = {0};

    if (create_device_buffer(size_A,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &buf_A) != 0) return -1;

    if (create_device_buffer(size_B,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &buf_B) != 0) {
        destroy_device_buffer(&buf_A);
        return -1;
    }

    if (create_device_buffer(size_C,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &buf_C) != 0) {
        destroy_device_buffer(&buf_A);
        destroy_device_buffer(&buf_B);
        return -1;
    }

    // Upload A and B
    void *map;
    vkMapMemory(vk_device, buf_A.memory, 0, VK_WHOLE_SIZE, 0, &map);
    memcpy(map, A_host, size_A);
    vkUnmapMemory(vk_device, buf_A.memory);

    vkMapMemory(vk_device, buf_B.memory, 0, VK_WHOLE_SIZE, 0, &map);
    memcpy(map, B_host, size_B);
    vkUnmapMemory(vk_device, buf_B.memory);

    // Update descriptor set
    VkDescriptorBufferInfo desc_A = { .buffer = buf_A.buffer, .offset = 0, .range = VK_WHOLE_SIZE };
    VkDescriptorBufferInfo desc_B = { .buffer = buf_B.buffer, .offset = 0, .range = VK_WHOLE_SIZE };
    VkDescriptorBufferInfo desc_C = { .buffer = buf_C.buffer, .offset = 0, .range = VK_WHOLE_SIZE };
    VkDescriptorBufferInfo desc_LUT = { .buffer = vk_lut_buffer, .offset = 0, .range = VK_WHOLE_SIZE };

    VkWriteDescriptorSet writes[4] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_desc_set,
          .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &desc_A },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_desc_set,
          .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &desc_B },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_desc_set,
          .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &desc_C },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_desc_set,
          .dstBinding = 3, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &desc_LUT },
    };
    vkUpdateDescriptorSets(vk_device, 4, writes, 0, NULL);

    // Create command buffer, dispatch, submit
    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk_cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(vk_device, &cmd_alloc, &cmd);

    VkCommandBufferBeginInfo cmd_begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &cmd_begin);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline_layout,
                            0, 1, &vk_desc_set, 0, NULL);

    struct { int M, N, K_div4; } pc = { M, N, K_div4 };
    vkCmdPushConstants(cmd, vk_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    int group_x = (M + 7) / 8;
    int group_y = (N + 7) / 8;
    vkCmdDispatch(cmd, (uint32_t)group_x, (uint32_t)group_y, 1);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    vkQueueSubmit(vk_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(vk_queue);

    vkFreeCommandBuffers(vk_device, vk_cmd_pool, 1, &cmd);

    // Read back C
    vkMapMemory(vk_device, buf_C.memory, 0, VK_WHOLE_SIZE, 0, &map);
    memcpy(C_host, map, size_C);
    vkUnmapMemory(vk_device, buf_C.memory);

    destroy_device_buffer(&buf_A);
    destroy_device_buffer(&buf_B);
    destroy_device_buffer(&buf_C);

    return 0;
}

// ============================================================
// CPU fallback (for verification / when Vulkan unavailable)
// ============================================================

void floatpack_matmul_fp4_cpu(int M, int N, int K_div4,
                              const uint16_t *A, const uint16_t *B, float *C) {
    float lut[FP4_LUT_SIZE];
    fp4_build_lut(lut);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K_div4; k++) {
                uint16_t a_word = A[i * K_div4 + k];
                uint16_t b_word = B[j * K_div4 + k];

                uint8_t a[4], b[4];
                fp4_unpack_4(a_word, a);
                fp4_unpack_4(b_word, b);

                for (int l = 0; l < 4; l++) {
                    acc += lut[(a[l] << 4) | b[l]];
                }
            }
            C[i * N + j] = acc;
        }
    }
}

// ============================================================
// Q4_0 -> FP4 converter (POC: quick integration with ggml-vulkan)
// ============================================================

// fp16 -> fp32 (IEEE 754-2008)
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        // zero / subnormal
        f = sign | (mant << 13);
    } else if (exp == 31) {
        // inf / nan
        f = sign | (0xFF << 23) | (mant << 13);
    } else {
        // normal
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

#define Q4K 32
// Mirror of ggml's block_q4_0
typedef struct {
    uint16_t d;    // fp16 scale
    uint8_t qs[Q4K / 2]; // nibbles
} fp_block_q4_0;

// Convert one row of Q4_0 blocks to FP4-packed uint16_t
// n_blocks = K / Q4K
static void convert_q4_row_to_fp4(const fp_block_q4_0 *blocks, int n_blocks, uint16_t *out) {
    for (int b = 0; b < n_blocks; b++) {
        float d = fp16_to_fp32(blocks[b].d);
        const uint8_t *qs = blocks[b].qs;
        for (int i = 0; i < Q4K; i += 4) {
            uint8_t nibbles[4];
            for (int j = 0; j < 4; j++) {
                int idx = i + j;
                int nib = (qs[idx >> 1] >> ((idx & 1) ? 4 : 0)) & 0xF;
                float val = d * ((float)(int)(nib - 8));
                nibbles[j] = fp4_encode(val);
            }
            out[b * (Q4K / 4) + (i / 4)] = fp4_pack_4(nibbles);
        }
    }
}

// Convert B matrix from F32 row-major (KxN) to FP4-packed column-major (N x K/4)
// B_f32[row * N + col]
// B_fp4[col * (K/4) + kword]
static void convert_f32_to_fp4_colmaj(const float *B_f32, int K, int N,
                                      uint16_t *B_fp4) {
    int K_div4 = K / 4;
    for (int col = 0; col < N; col++) {
        for (int k = 0; k < K; k += 4) {
            uint8_t nibbles[4];
            for (int j = 0; j < 4; j++) {
                nibbles[j] = fp4_encode(B_f32[(k + j) * N + col]);
            }
            B_fp4[col * K_div4 + (k / 4)] = fp4_pack_4(nibbles);
        }
    }
}

// Convert A matrix from F32 row-major (MxK) to FP4-packed row-major (M x K/4)
// A_f32[row * K + col]
// A_fp4[row * (K/4) + kword]
static void convert_f32_to_fp4_rowmaj(const float *A_f32, int M, int K,
                                       uint16_t *A_fp4) {
    int K_div4 = K / 4;
    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k += 4) {
            uint8_t nibbles[4];
            for (int j = 0; j < 4; j++) {
                nibbles[j] = fp4_encode(A_f32[row * K + k + j]);
            }
            A_fp4[row * K_div4 + (k / 4)] = fp4_pack_4(nibbles);
        }
    }
}

int floatpack_matmul_f32(int M, int N, int K,
                         const float *A_f32, const float *B_f32, float *C_f32) {
    const int K_div4 = K / 4;
    if (K_div4 <= 0 || K % 4 != 0) return -1;

    uint16_t *A_fp4 = (uint16_t *)malloc((size_t)M * K_div4 * sizeof(uint16_t));
    uint16_t *B_fp4 = (uint16_t *)malloc((size_t)N * K_div4 * sizeof(uint16_t));
    if (!A_fp4 || !B_fp4) { free(A_fp4); free(B_fp4); return -1; }

    convert_f32_to_fp4_rowmaj(A_f32, M, K, A_fp4);
    convert_f32_to_fp4_colmaj(B_f32, K, N, B_fp4);

    int ret;
    if (vk_initialized) {
        ret = floatpack_matmul_fp4(M, N, K_div4, A_fp4, B_fp4, C_f32);
    } else {
        floatpack_matmul_fp4_cpu(M, N, K_div4, A_fp4, B_fp4, C_f32);
        ret = 0;
    }

    free(A_fp4);
    free(B_fp4);
    return ret;
}

int floatpack_matmul_q4_0(int M, int N, int K,
                          const void *A_q4, const float *B_f32, float *C_f32) {
    const int K_div4 = K / 4;
    const int n_blocks = K / Q4K;

    // allocate temp host buffers
    uint16_t *A_fp4 = (uint16_t *)malloc((size_t)M * K_div4 * sizeof(uint16_t));
    uint16_t *B_fp4 = (uint16_t *)malloc((size_t)N * K_div4 * sizeof(uint16_t));
    if (!A_fp4 || !B_fp4) { free(A_fp4); free(B_fp4); return -1; }

    // convert A: Q4_0 -> FP4-packed
    for (int i = 0; i < M; i++) {
        const fp_block_q4_0 *row_blocks = (const fp_block_q4_0 *)A_q4 + i * n_blocks;
        convert_q4_row_to_fp4(row_blocks, n_blocks, A_fp4 + i * K_div4);
    }

    // convert B: F32 row-major -> FP4-packed column-major
    convert_f32_to_fp4_colmaj(B_f32, K, N, B_fp4);

    int ret;
    if (vk_initialized) {
        ret = floatpack_matmul_fp4(M, N, K_div4, A_fp4, B_fp4, C_f32);
    } else {
        floatpack_matmul_fp4_cpu(M, N, K_div4, A_fp4, B_fp4, C_f32);
        ret = 0;
    }

    free(A_fp4);
    free(B_fp4);
    return ret;
}

// ============================================================
// Fused Q4_0 matmul (no intermediate FP4 conversion)
// ============================================================

int floatpack_matmul_q4_0_fused(int M, int N, int K,
                                const void *A_q4, const float *B_f32, float *C_f32) {
    if (!vk_initialized) {
        // CPU fallback: dequantize A to F32, then do standard F32 matmul
        int K_blocks = K / 32;
        if (K_blocks <= 0 || K % 32 != 0) return -1;
        float *A_f32 = (float *)malloc((size_t)M * K * sizeof(float));
        if (!A_f32) return -1;
        const uint16_t *d_src = (const uint16_t *)A_q4; // scales (fp16) at start of each block
        const uint8_t  *q_src = (const uint8_t  *)A_q4; // nibbles start at offset 2
        for (int i = 0; i < M; i++) {
            for (int kb = 0; kb < K_blocks; kb++) {
                int blk_off = (i * K_blocks + kb) * 18; // byte offset
                uint16_t d_bits;
                memcpy(&d_bits, (const uint8_t *)A_q4 + blk_off, 2);
                float d = fp16_to_fp32(d_bits);
                const uint8_t *qs = (const uint8_t *)A_q4 + blk_off + 2;
                for (int j = 0; j < 32; j++) {
                    int nib = (qs[j >> 1] >> ((j & 1) ? 4 : 0)) & 0xF;
                    A_f32[i * K + kb * 32 + j] = d * ((float)(nib - 8));
                }
            }
        }
        // F32 matmul: C = A * B (A: MxK, B: KxN row-major)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    acc += A_f32[i * K + k] * B_f32[k * N + j];
                }
                C_f32[i * N + j] = acc;
            }
        }
        free(A_f32);
        return 0;
    }

    int K_blocks = K / 32;
    if (K_blocks <= 0 || K % 32 != 0) return -1;

    size_t size_A = (size_t)M * K_blocks * 18; // Q4_0 blocks (bytes)
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    fp_vk_buffer buf_A = {0}, buf_B = {0}, buf_C = {0};

    if (create_device_buffer(size_A,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &buf_A) != 0) return -1;

    if (create_device_buffer(size_B,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &buf_B) != 0) {
        destroy_device_buffer(&buf_A);
        return -1;
    }

    if (create_device_buffer(size_C,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &buf_C) != 0) {
        destroy_device_buffer(&buf_A);
        destroy_device_buffer(&buf_B);
        return -1;
    }

    // Upload A and B
    void *map;
    vkMapMemory(vk_device, buf_A.memory, 0, VK_WHOLE_SIZE, 0, &map);
    memcpy(map, A_q4, size_A);
    vkUnmapMemory(vk_device, buf_A.memory);

    vkMapMemory(vk_device, buf_B.memory, 0, VK_WHOLE_SIZE, 0, &map);
    memcpy(map, B_f32, size_B);
    vkUnmapMemory(vk_device, buf_B.memory);

    // Update descriptor set (bindings 0,1,2; skip binding 3)
    VkDescriptorBufferInfo desc_A = { .buffer = buf_A.buffer, .offset = 0, .range = VK_WHOLE_SIZE };
    VkDescriptorBufferInfo desc_B = { .buffer = buf_B.buffer, .offset = 0, .range = VK_WHOLE_SIZE };
    VkDescriptorBufferInfo desc_C = { .buffer = buf_C.buffer, .offset = 0, .range = VK_WHOLE_SIZE };

    VkWriteDescriptorSet writes[3] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_desc_set,
          .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &desc_A },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_desc_set,
          .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &desc_B },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_desc_set,
          .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &desc_C },
    };
    vkUpdateDescriptorSets(vk_device, 3, writes, 0, NULL);

    // Create command buffer, dispatch, submit
    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk_cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(vk_device, &cmd_alloc, &cmd);

    VkCommandBufferBeginInfo cmd_begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &cmd_begin);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline_q4);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline_layout,
                            0, 1, &vk_desc_set, 0, NULL);

    struct { int M, N, K_blocks; } pc = { M, N, K_blocks };
    vkCmdPushConstants(cmd, vk_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    int group_x = (M + 7) / 8;
    int group_y = (N + 7) / 8;
    vkCmdDispatch(cmd, (uint32_t)group_x, (uint32_t)group_y, 1);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    vkQueueSubmit(vk_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(vk_queue);

    vkFreeCommandBuffers(vk_device, vk_cmd_pool, 1, &cmd);

    // Read back C
    vkMapMemory(vk_device, buf_C.memory, 0, VK_WHOLE_SIZE, 0, &map);
    memcpy(C_f32, map, size_C);
    vkUnmapMemory(vk_device, buf_C.memory);

    destroy_device_buffer(&buf_A);
    destroy_device_buffer(&buf_B);
    destroy_device_buffer(&buf_C);

    return 0;
}
