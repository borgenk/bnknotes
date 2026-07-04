//! The Vulkan backend's device layer: the slice of the Vulkan C ABI the GPU
//! presentation path uses, and the bring-up that turns the compositor's dmabuf
//! feedback into a live device and queue.
//!
//! The loader is `dlopen`ed rather than linked, so a machine without Vulkan
//! falls back to software rendering instead of failing to start; every
//! function pointer is fetched through `vkGetInstanceProcAddr` /
//! `vkGetDeviceProcAddr` as the spec prescribes. There is no window surface
//! and no swapchain anywhere in this module: presentation is dmabuf export
//! over the Wayland wire, so the WSI machinery never appears.
//!
//! Transcription conventions: struct and constant names keep Vulkan's own
//! (minus the `Vk`/`VK_` prefix would hurt grep-ability, so they keep it);
//! fields are renamed to snake_case; only the structure types, results, and
//! flags actually exercised are declared. Layouts mirror the x86_64/arm64 C
//! ABI, guarded by size assertions in the tests at the bottom.

use core::ffi::{c_char, c_void, CStr};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

use crate::error::{Error, Result};
use crate::ffi::DynLib;
use crate::gpu::{FrameData, ImageKey, Vertex};

// ---------------------------------------------------------------------------
// Handles, scalars, and constants.
// ---------------------------------------------------------------------------

/// Dispatchable handles are pointers.
pub type VkInstance = *mut c_void;
pub type VkPhysicalDevice = *mut c_void;
pub type VkDevice = *mut c_void;
pub type VkQueue = *mut c_void;
pub type VkCommandBuffer = *mut c_void;

/// Non-dispatchable handles are 64-bit opaque values.
type VkImage = u64;
type VkDeviceMemory = u64;
type VkSemaphore = u64;
type VkFence = u64;
type VkCommandPool = u64;
type VkBuffer = u64;
type VkImageView = u64;
type VkFramebuffer = u64;
type VkRenderPass = u64;
type VkPipeline = u64;
type VkPipelineLayout = u64;
type VkShaderModule = u64;
type VkDescriptorSetLayout = u64;
type VkDescriptorPool = u64;
type VkDescriptorSet = u64;
type VkSampler = u64;

type VkResult = i32;
const VK_SUCCESS: VkResult = 0;
/// The enumerate-style calls return this when the caller's array was smaller
/// than the full set; the two-call pattern below retries.
const VK_INCOMPLETE: VkResult = 5;

/// `VK_API_VERSION_1_2`: high enough that every dependency of the extensions
/// this backend enables is core (bind_memory2, image_format_list, external
/// memory/semaphore capabilities), low enough for any 2020+ driver.
const API_VERSION_1_2: u32 = (1 << 22) | (2 << 12);

const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
const VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO: u32 = 9;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
const VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO: u32 = 14;
const VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO: u32 = 15;
const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
const VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO: u32 = 19;
const VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO: u32 = 20;
const VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO: u32 = 22;
const VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO: u32 = 23;
const VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO: u32 = 24;
const VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO: u32 = 26;
const VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO: u32 = 27;
const VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO: u32 = 28;
const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
const VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO: u32 = 31;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
const VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO: u32 = 37;
const VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO: u32 = 38;
const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
const VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO: u32 = 43;
const VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER: u32 = 45;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2: u32 = 1000059001;
const VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2: u32 = 1000059002;
const VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO: u32 = 1000072001;
const VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO: u32 = 1000072002;
const VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR: u32 = 1000074002;
const VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO: u32 = 1000077000;
const VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR: u32 = 1000079000;
const VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR: u32 = 1000079001;
const VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO: u32 = 1000127001;
const VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT: u32 = 1000158000;
const VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT: u32 = 1000158003;
const VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT: u32 = 1000158005;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT: u32 = 1000353000;

/// `VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT`.
const VK_QUEUE_GRAPHICS_BIT: u32 = 0x1;

/// `VK_FORMAT_B8G8R8A8_UNORM`: bytes B,G,R,A in memory, the Vulkan twin of
/// little-endian DRM XRGB8888 (and of the software canvas's pixel layout).
const VK_FORMAT_B8G8R8A8_UNORM: u32 = 44;
/// `VK_FORMAT_R8_UNORM`: the glyph coverage atlas.
const VK_FORMAT_R8_UNORM: u32 = 9;
/// Vertex attribute formats.
const VK_FORMAT_R32_UINT: u32 = 98;
const VK_FORMAT_R32G32_SFLOAT: u32 = 103;
const VK_FORMAT_R32G32B32A32_SFLOAT: u32 = 109;

/// `VkImageTiling::VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT`.
const VK_IMAGE_TILING_DRM_FORMAT_MODIFIER: u32 = 1000158000;
const VK_IMAGE_TILING_OPTIMAL: u32 = 0;
const VK_IMAGE_TYPE_2D: u32 = 1;
const VK_IMAGE_VIEW_TYPE_2D: u32 = 1;
const VK_SAMPLE_COUNT_1_BIT: u32 = 1;
const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;

const VK_IMAGE_USAGE_TRANSFER_SRC_BIT: u32 = 0x1;
const VK_IMAGE_USAGE_TRANSFER_DST_BIT: u32 = 0x2;
const VK_IMAGE_USAGE_SAMPLED_BIT: u32 = 0x4;
const VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT: u32 = 0x10;
/// The format features a presentable image's modifier must support: rendering
/// plus both transfer directions (the pixel-parity readback uses the source
/// direction; clears use the destination).
const VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT: u32 = 0x80;
const VK_FORMAT_FEATURE_TRANSFER_SRC_BIT: u32 = 0x4000;
const VK_FORMAT_FEATURE_TRANSFER_DST_BIT: u32 = 0x8000;

const VK_IMAGE_LAYOUT_UNDEFINED: u32 = 0;
/// The layout the image is handed to the compositor in: the only layout
/// defined for foreign (non-Vulkan) access.
const VK_IMAGE_LAYOUT_GENERAL: u32 = 1;
const VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: u32 = 2;
const VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: u32 = 5;
const VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: u32 = 6;
const VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: u32 = 7;

const VK_ACCESS_SHADER_READ_BIT: u32 = 0x20;
const VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT: u32 = 0x100;
const VK_ACCESS_TRANSFER_READ_BIT: u32 = 0x800;
const VK_ACCESS_TRANSFER_WRITE_BIT: u32 = 0x1000;
const VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT: u32 = 0x1;
const VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT: u32 = 0x80;
const VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT: u32 = 0x400;
const VK_PIPELINE_STAGE_TRANSFER_BIT: u32 = 0x1000;
const VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT: u32 = 0x2000;

const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x1;
const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x2;
const VK_BUFFER_USAGE_VERTEX_BUFFER_BIT: u32 = 0x80;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x2;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x4;

const VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: u32 = 1;
const VK_SHADER_STAGE_VERTEX_BIT: u32 = 0x1;
const VK_SHADER_STAGE_FRAGMENT_BIT: u32 = 0x10;
const VK_PIPELINE_BIND_POINT_GRAPHICS: u32 = 0;
const VK_ATTACHMENT_LOAD_OP_CLEAR: u32 = 1;
const VK_ATTACHMENT_LOAD_OP_DONT_CARE: u32 = 2;
const VK_ATTACHMENT_STORE_OP_STORE: u32 = 0;
const VK_ATTACHMENT_STORE_OP_DONT_CARE: u32 = 1;
const VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: u32 = 3;
const VK_POLYGON_MODE_FILL: u32 = 0;
const VK_DYNAMIC_STATE_VIEWPORT: u32 = 0;
const VK_DYNAMIC_STATE_SCISSOR: u32 = 1;
const VK_BLEND_FACTOR_ZERO: u32 = 0;
const VK_BLEND_FACTOR_ONE: u32 = 1;
const VK_BLEND_FACTOR_SRC_ALPHA: u32 = 6;
const VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: u32 = 7;
const VK_BLEND_OP_ADD: u32 = 0;
const VK_COLOR_COMPONENT_RGBA: u32 = 0xf;
const VK_FILTER_NEAREST: u32 = 0;
const VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: u32 = 2;
const VK_VERTEX_INPUT_RATE_VERTEX: u32 = 0;
const VK_SUBPASS_CONTENTS_INLINE: u32 = 0;
const VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT: u32 = 0x1;
/// Room for the atlas set, the dummy set, and a generous number of distinct
/// visible images; a notes document with more is out of scope.
const DESCRIPTOR_MAX_SETS: u32 = 256;

const VK_IMAGE_ASPECT_COLOR_BIT: u32 = 0x1;
/// The dmabuf plane aspect for layout queries under modifier tiling.
const VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT: u32 = 0x80;

const VK_QUEUE_FAMILY_IGNORED: u32 = u32::MAX;
/// `VK_QUEUE_FAMILY_FOREIGN_EXT`: the compositor's side of the ownership
/// transfer barriers around each frame.
const VK_QUEUE_FAMILY_FOREIGN: u32 = u32::MAX - 2;

const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x1;
const VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT: u32 = 0x200;
const VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT: u32 = 0x10;
/// `VK_SEMAPHORE_IMPORT_TEMPORARY_BIT`: a sync-fd import replaces the payload
/// for one wait, then the semaphore reverts, exactly the per-frame acquire.
const VK_SEMAPHORE_IMPORT_TEMPORARY_BIT: u32 = 0x1;

const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x2;
const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x1;

/// Every fence wait in this backend bounds at one second: a GPU that takes
/// longer to clear or paint a window is broken, and a clean error (falling
/// back to software) beats a frozen editor.
const FENCE_TIMEOUT_NS: u64 = 1_000_000_000;

/// `DRM_FORMAT_MOD_INVALID`: "the layout is implicit"; never allocate with it.
const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;
const VK_UUID_SIZE: usize = 16;
const VK_MAX_EXTENSION_NAME_SIZE: usize = 256;

/// The device extensions the dmabuf presentation path enables: image layouts
/// by DRM modifier, dmabuf memory export, fd-based semaphore export/import for
/// the implicit-sync bridge, and foreign-queue ownership transfer so the
/// compositor's reads are well-defined.
const REQUIRED_DEVICE_EXTENSIONS: [&CStr; 5] = [
    c"VK_EXT_image_drm_format_modifier",
    c"VK_EXT_external_memory_dma_buf",
    c"VK_KHR_external_memory_fd",
    c"VK_KHR_external_semaphore_fd",
    c"VK_EXT_queue_family_foreign",
];
/// Needed only to *identify* the device (its DRM dev_t); physical-device-level
/// properties may be queried when supported without enabling anything.
const EXT_PHYSICAL_DEVICE_DRM: &CStr = c"VK_EXT_physical_device_drm";

// ---------------------------------------------------------------------------
// Struct layouts (x86_64/arm64 C ABI).
// ---------------------------------------------------------------------------

#[repr(C)]
struct VkApplicationInfo {
    s_type: u32,
    p_next: *const c_void,
    p_application_name: *const c_char,
    application_version: u32,
    p_engine_name: *const c_char,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
struct VkInstanceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    p_application_info: *const VkApplicationInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const c_char,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const c_char,
}

#[repr(C)]
struct VkDeviceQueueCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_family_index: u32,
    queue_count: u32,
    p_queue_priorities: *const f32,
}

#[repr(C)]
struct VkDeviceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_create_info_count: u32,
    p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const c_char,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const c_char,
    p_enabled_features: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkExtensionProperties {
    extension_name: [c_char; VK_MAX_EXTENSION_NAME_SIZE],
    spec_version: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkExtent3D {
    width: u32,
    height: u32,
    depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkQueueFamilyProperties {
    queue_flags: u32,
    queue_count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: VkExtent3D,
}

/// `VkPhysicalDeviceLimits`, transcribed in full: the properties struct embeds
/// it by value, so its exact size is load-bearing for everything after it on
/// the stack. The size assertion in the tests guards the transcription.
#[repr(C)]
struct VkPhysicalDeviceLimits {
    max_image_dimension_1d: u32,
    max_image_dimension_2d: u32,
    max_image_dimension_3d: u32,
    max_image_dimension_cube: u32,
    max_image_array_layers: u32,
    max_texel_buffer_elements: u32,
    max_uniform_buffer_range: u32,
    max_storage_buffer_range: u32,
    max_push_constants_size: u32,
    max_memory_allocation_count: u32,
    max_sampler_allocation_count: u32,
    buffer_image_granularity: u64,
    sparse_address_space_size: u64,
    max_bound_descriptor_sets: u32,
    max_per_stage_descriptor_samplers: u32,
    max_per_stage_descriptor_uniform_buffers: u32,
    max_per_stage_descriptor_storage_buffers: u32,
    max_per_stage_descriptor_sampled_images: u32,
    max_per_stage_descriptor_storage_images: u32,
    max_per_stage_descriptor_input_attachments: u32,
    max_per_stage_resources: u32,
    max_descriptor_set_samplers: u32,
    max_descriptor_set_uniform_buffers: u32,
    max_descriptor_set_uniform_buffers_dynamic: u32,
    max_descriptor_set_storage_buffers: u32,
    max_descriptor_set_storage_buffers_dynamic: u32,
    max_descriptor_set_sampled_images: u32,
    max_descriptor_set_storage_images: u32,
    max_descriptor_set_input_attachments: u32,
    max_vertex_input_attributes: u32,
    max_vertex_input_bindings: u32,
    max_vertex_input_attribute_offset: u32,
    max_vertex_input_binding_stride: u32,
    max_vertex_output_components: u32,
    max_tessellation_generation_level: u32,
    max_tessellation_patch_size: u32,
    max_tessellation_control_per_vertex_input_components: u32,
    max_tessellation_control_per_vertex_output_components: u32,
    max_tessellation_control_per_patch_output_components: u32,
    max_tessellation_control_total_output_components: u32,
    max_tessellation_evaluation_input_components: u32,
    max_tessellation_evaluation_output_components: u32,
    max_geometry_shader_invocations: u32,
    max_geometry_input_components: u32,
    max_geometry_output_components: u32,
    max_geometry_output_vertices: u32,
    max_geometry_total_output_components: u32,
    max_fragment_input_components: u32,
    max_fragment_output_attachments: u32,
    max_fragment_dual_src_attachments: u32,
    max_fragment_combined_output_resources: u32,
    max_compute_shared_memory_size: u32,
    max_compute_work_group_count: [u32; 3],
    max_compute_work_group_invocations: u32,
    max_compute_work_group_size: [u32; 3],
    sub_pixel_precision_bits: u32,
    sub_texel_precision_bits: u32,
    mipmap_precision_bits: u32,
    max_draw_indexed_index_value: u32,
    max_draw_indirect_count: u32,
    max_sampler_lod_bias: f32,
    max_sampler_anisotropy: f32,
    max_viewports: u32,
    max_viewport_dimensions: [u32; 2],
    viewport_bounds_range: [f32; 2],
    viewport_sub_pixel_bits: u32,
    min_memory_map_alignment: usize,
    min_texel_buffer_offset_alignment: u64,
    min_uniform_buffer_offset_alignment: u64,
    min_storage_buffer_offset_alignment: u64,
    min_texel_offset: i32,
    max_texel_offset: u32,
    min_texel_gather_offset: i32,
    max_texel_gather_offset: u32,
    min_interpolation_offset: f32,
    max_interpolation_offset: f32,
    sub_pixel_interpolation_offset_bits: u32,
    max_framebuffer_width: u32,
    max_framebuffer_height: u32,
    max_framebuffer_layers: u32,
    framebuffer_color_sample_counts: u32,
    framebuffer_depth_sample_counts: u32,
    framebuffer_stencil_sample_counts: u32,
    framebuffer_no_attachments_sample_counts: u32,
    max_color_attachments: u32,
    sampled_image_color_sample_counts: u32,
    sampled_image_integer_sample_counts: u32,
    sampled_image_depth_sample_counts: u32,
    sampled_image_stencil_sample_counts: u32,
    storage_image_sample_counts: u32,
    max_sample_mask_words: u32,
    timestamp_compute_and_graphics: u32,
    timestamp_period: f32,
    max_clip_distances: u32,
    max_cull_distances: u32,
    max_combined_clip_and_cull_distances: u32,
    discrete_queue_priorities: u32,
    point_size_range: [f32; 2],
    line_width_range: [f32; 2],
    point_size_granularity: f32,
    line_width_granularity: f32,
    strict_lines: u32,
    standard_sample_locations: u32,
    optimal_buffer_copy_offset_alignment: u64,
    optimal_buffer_copy_row_pitch_alignment: u64,
    non_coherent_atom_size: u64,
}

#[repr(C)]
struct VkPhysicalDeviceSparseProperties {
    residency_standard_2d_block_shape: u32,
    residency_standard_2d_multisample_block_shape: u32,
    residency_standard_3d_block_shape: u32,
    residency_aligned_mip_size: u32,
    residency_non_resident_strict: u32,
}

#[repr(C)]
struct VkPhysicalDeviceProperties {
    api_version: u32,
    driver_version: u32,
    vendor_id: u32,
    device_id: u32,
    device_type: u32,
    device_name: [c_char; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE],
    pipeline_cache_uuid: [u8; VK_UUID_SIZE],
    limits: VkPhysicalDeviceLimits,
    sparse_properties: VkPhysicalDeviceSparseProperties,
}

#[repr(C)]
struct VkPhysicalDeviceProperties2 {
    s_type: u32,
    p_next: *mut c_void,
    properties: VkPhysicalDeviceProperties,
}

/// `VK_EXT_physical_device_drm`: which DRM primary/render node a physical
/// device is, matched against the dev_t in the compositor's dmabuf feedback.
#[repr(C)]
struct VkPhysicalDeviceDrmPropertiesEXT {
    s_type: u32,
    p_next: *mut c_void,
    has_primary: u32,
    has_render: u32,
    primary_major: i64,
    primary_minor: i64,
    render_major: i64,
    render_minor: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryType {
    property_flags: u32,
    heap_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkMemoryHeap {
    size: u64,
    flags: u32,
}

#[repr(C)]
struct VkPhysicalDeviceMemoryProperties {
    memory_type_count: u32,
    memory_types: [VkMemoryType; 32],
    memory_heap_count: u32,
    memory_heaps: [VkMemoryHeap; 16],
}

#[repr(C)]
struct VkFormatProperties {
    linear_tiling_features: u32,
    optimal_tiling_features: u32,
    buffer_features: u32,
}

#[repr(C)]
struct VkFormatProperties2 {
    s_type: u32,
    p_next: *mut c_void,
    format_properties: VkFormatProperties,
}

/// One modifier the device supports for a format: its layout, how many memory
/// planes it uses, and what the format can do under it.
#[repr(C)]
#[derive(Clone, Copy)]
struct VkDrmFormatModifierPropertiesEXT {
    drm_format_modifier: u64,
    drm_format_modifier_plane_count: u32,
    drm_format_modifier_tiling_features: u32,
}

#[repr(C)]
struct VkDrmFormatModifierPropertiesListEXT {
    s_type: u32,
    p_next: *mut c_void,
    drm_format_modifier_count: u32,
    p_drm_format_modifier_properties: *mut VkDrmFormatModifierPropertiesEXT,
}

#[repr(C)]
struct VkImageCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    image_type: u32,
    format: u32,
    extent: VkExtent3D,
    mip_levels: u32,
    array_layers: u32,
    samples: u32,
    tiling: u32,
    usage: u32,
    sharing_mode: u32,
    queue_family_index_count: u32,
    p_queue_family_indices: *const u32,
    initial_layout: u32,
}

#[repr(C)]
struct VkExternalMemoryImageCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    handle_types: u32,
}

#[repr(C)]
struct VkImageDrmFormatModifierListCreateInfoEXT {
    s_type: u32,
    p_next: *const c_void,
    drm_format_modifier_count: u32,
    p_drm_format_modifiers: *const u64,
}

#[repr(C)]
struct VkImageDrmFormatModifierPropertiesEXT {
    s_type: u32,
    p_next: *mut c_void,
    drm_format_modifier: u64,
}

#[repr(C)]
struct VkMemoryRequirements {
    size: u64,
    alignment: u64,
    memory_type_bits: u32,
}

#[repr(C)]
struct VkMemoryAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    allocation_size: u64,
    memory_type_index: u32,
}

#[repr(C)]
struct VkExportMemoryAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    handle_types: u32,
}

/// Exportable images get a dedicated allocation: it is what external-memory
/// consumers expect, and it makes the exported fd describe exactly one image.
#[repr(C)]
struct VkMemoryDedicatedAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    image: VkImage,
    buffer: u64,
}

#[repr(C)]
struct VkMemoryGetFdInfoKHR {
    s_type: u32,
    p_next: *const c_void,
    memory: VkDeviceMemory,
    handle_type: u32,
}

#[repr(C)]
struct VkImageSubresource {
    aspect_mask: u32,
    mip_level: u32,
    array_layer: u32,
}

#[repr(C)]
struct VkSubresourceLayout {
    offset: u64,
    size: u64,
    row_pitch: u64,
    array_pitch: u64,
    depth_pitch: u64,
}

#[repr(C)]
struct VkCommandPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_family_index: u32,
}

#[repr(C)]
struct VkCommandBufferAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    command_pool: VkCommandPool,
    level: u32,
    command_buffer_count: u32,
}

#[repr(C)]
struct VkCommandBufferBeginInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    p_inheritance_info: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkImageSubresourceRange {
    aspect_mask: u32,
    base_mip_level: u32,
    level_count: u32,
    base_array_layer: u32,
    layer_count: u32,
}

#[repr(C)]
struct VkImageMemoryBarrier {
    s_type: u32,
    p_next: *const c_void,
    src_access_mask: u32,
    dst_access_mask: u32,
    old_layout: u32,
    new_layout: u32,
    src_queue_family_index: u32,
    dst_queue_family_index: u32,
    image: VkImage,
    subresource_range: VkImageSubresourceRange,
}

#[repr(C)]
struct VkSubmitInfo {
    s_type: u32,
    p_next: *const c_void,
    wait_semaphore_count: u32,
    p_wait_semaphores: *const VkSemaphore,
    p_wait_dst_stage_mask: *const u32,
    command_buffer_count: u32,
    p_command_buffers: *const VkCommandBuffer,
    signal_semaphore_count: u32,
    p_signal_semaphores: *const VkSemaphore,
}

#[repr(C)]
struct VkFenceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
}

#[repr(C)]
struct VkSemaphoreCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
}

#[repr(C)]
struct VkExportSemaphoreCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    handle_types: u32,
}

#[repr(C)]
struct VkSemaphoreGetFdInfoKHR {
    s_type: u32,
    p_next: *const c_void,
    semaphore: VkSemaphore,
    handle_type: u32,
}

#[repr(C)]
struct VkImportSemaphoreFdInfoKHR {
    s_type: u32,
    p_next: *const c_void,
    semaphore: VkSemaphore,
    flags: u32,
    handle_type: u32,
    fd: i32,
}

/// `VkClearColorValue`, always used as float32 here.
#[repr(C)]
struct VkClearColorValue {
    float32: [f32; 4],
}

#[repr(C)]
struct VkBufferCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    size: u64,
    usage: u32,
    sharing_mode: u32,
    queue_family_index_count: u32,
    p_queue_family_indices: *const u32,
}

#[repr(C)]
struct VkComponentMapping {
    r: u32,
    g: u32,
    b: u32,
    a: u32,
}

#[repr(C)]
struct VkImageViewCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    image: VkImage,
    view_type: u32,
    format: u32,
    components: VkComponentMapping,
    subresource_range: VkImageSubresourceRange,
}

#[repr(C)]
struct VkShaderModuleCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    code_size: usize,
    p_code: *const u32,
}

#[repr(C)]
struct VkPipelineShaderStageCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: u32,
    module: VkShaderModule,
    p_name: *const c_char,
    p_specialization_info: *const c_void,
}

#[repr(C)]
struct VkVertexInputBindingDescription {
    binding: u32,
    stride: u32,
    input_rate: u32,
}

#[repr(C)]
struct VkVertexInputAttributeDescription {
    location: u32,
    binding: u32,
    format: u32,
    offset: u32,
}

#[repr(C)]
struct VkPipelineVertexInputStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    vertex_binding_description_count: u32,
    p_vertex_binding_descriptions: *const VkVertexInputBindingDescription,
    vertex_attribute_description_count: u32,
    p_vertex_attribute_descriptions: *const VkVertexInputAttributeDescription,
}

#[repr(C)]
struct VkPipelineInputAssemblyStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    topology: u32,
    primitive_restart_enable: u32,
}

#[repr(C)]
struct VkViewport {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkOffset2D {
    x: i32,
    y: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkExtent2D {
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkRect2D {
    offset: VkOffset2D,
    extent: VkExtent2D,
}

#[repr(C)]
struct VkPipelineViewportStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    viewport_count: u32,
    p_viewports: *const VkViewport,
    scissor_count: u32,
    p_scissors: *const VkRect2D,
}

#[repr(C)]
struct VkPipelineRasterizationStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    depth_clamp_enable: u32,
    rasterizer_discard_enable: u32,
    polygon_mode: u32,
    cull_mode: u32,
    front_face: u32,
    depth_bias_enable: u32,
    depth_bias_constant_factor: f32,
    depth_bias_clamp: f32,
    depth_bias_slope_factor: f32,
    line_width: f32,
}

#[repr(C)]
struct VkPipelineMultisampleStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    rasterization_samples: u32,
    sample_shading_enable: u32,
    min_sample_shading: f32,
    p_sample_mask: *const u32,
    alpha_to_coverage_enable: u32,
    alpha_to_one_enable: u32,
}

#[repr(C)]
struct VkPipelineColorBlendAttachmentState {
    blend_enable: u32,
    src_color_blend_factor: u32,
    dst_color_blend_factor: u32,
    color_blend_op: u32,
    src_alpha_blend_factor: u32,
    dst_alpha_blend_factor: u32,
    alpha_blend_op: u32,
    color_write_mask: u32,
}

#[repr(C)]
struct VkPipelineColorBlendStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    logic_op_enable: u32,
    logic_op: u32,
    attachment_count: u32,
    p_attachments: *const VkPipelineColorBlendAttachmentState,
    blend_constants: [f32; 4],
}

#[repr(C)]
struct VkPipelineDynamicStateCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    dynamic_state_count: u32,
    p_dynamic_states: *const u32,
}

#[repr(C)]
struct VkPushConstantRange {
    stage_flags: u32,
    offset: u32,
    size: u32,
}

#[repr(C)]
struct VkPipelineLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    set_layout_count: u32,
    p_set_layouts: *const VkDescriptorSetLayout,
    push_constant_range_count: u32,
    p_push_constant_ranges: *const VkPushConstantRange,
}

#[repr(C)]
struct VkGraphicsPipelineCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage_count: u32,
    p_stages: *const VkPipelineShaderStageCreateInfo,
    p_vertex_input_state: *const VkPipelineVertexInputStateCreateInfo,
    p_input_assembly_state: *const VkPipelineInputAssemblyStateCreateInfo,
    p_tessellation_state: *const c_void,
    p_viewport_state: *const VkPipelineViewportStateCreateInfo,
    p_rasterization_state: *const VkPipelineRasterizationStateCreateInfo,
    p_multisample_state: *const VkPipelineMultisampleStateCreateInfo,
    p_depth_stencil_state: *const c_void,
    p_color_blend_state: *const VkPipelineColorBlendStateCreateInfo,
    p_dynamic_state: *const VkPipelineDynamicStateCreateInfo,
    layout: VkPipelineLayout,
    render_pass: VkRenderPass,
    subpass: u32,
    base_pipeline_handle: VkPipeline,
    base_pipeline_index: i32,
}

#[repr(C)]
struct VkSamplerCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    mag_filter: u32,
    min_filter: u32,
    mipmap_mode: u32,
    address_mode_u: u32,
    address_mode_v: u32,
    address_mode_w: u32,
    mip_lod_bias: f32,
    anisotropy_enable: u32,
    max_anisotropy: f32,
    compare_enable: u32,
    compare_op: u32,
    min_lod: f32,
    max_lod: f32,
    border_color: u32,
    unnormalized_coordinates: u32,
}

#[repr(C)]
struct VkDescriptorSetLayoutBinding {
    binding: u32,
    descriptor_type: u32,
    descriptor_count: u32,
    stage_flags: u32,
    p_immutable_samplers: *const VkSampler,
}

#[repr(C)]
struct VkDescriptorSetLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    binding_count: u32,
    p_bindings: *const VkDescriptorSetLayoutBinding,
}

#[repr(C)]
struct VkDescriptorPoolSize {
    descriptor_type: u32,
    descriptor_count: u32,
}

#[repr(C)]
struct VkDescriptorPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    max_sets: u32,
    pool_size_count: u32,
    p_pool_sizes: *const VkDescriptorPoolSize,
}

#[repr(C)]
struct VkDescriptorSetAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    descriptor_pool: VkDescriptorPool,
    descriptor_set_count: u32,
    p_set_layouts: *const VkDescriptorSetLayout,
}

#[repr(C)]
struct VkDescriptorImageInfo {
    sampler: VkSampler,
    image_view: VkImageView,
    image_layout: u32,
}

#[repr(C)]
struct VkWriteDescriptorSet {
    s_type: u32,
    p_next: *const c_void,
    dst_set: VkDescriptorSet,
    dst_binding: u32,
    dst_array_element: u32,
    descriptor_count: u32,
    descriptor_type: u32,
    p_image_info: *const VkDescriptorImageInfo,
    p_buffer_info: *const c_void,
    p_texel_buffer_view: *const c_void,
}

#[repr(C)]
struct VkAttachmentDescription {
    flags: u32,
    format: u32,
    samples: u32,
    load_op: u32,
    store_op: u32,
    stencil_load_op: u32,
    stencil_store_op: u32,
    initial_layout: u32,
    final_layout: u32,
}

#[repr(C)]
struct VkAttachmentReference {
    attachment: u32,
    layout: u32,
}

#[repr(C)]
struct VkSubpassDescription {
    flags: u32,
    pipeline_bind_point: u32,
    input_attachment_count: u32,
    p_input_attachments: *const VkAttachmentReference,
    color_attachment_count: u32,
    p_color_attachments: *const VkAttachmentReference,
    p_resolve_attachments: *const VkAttachmentReference,
    p_depth_stencil_attachment: *const VkAttachmentReference,
    preserve_attachment_count: u32,
    p_preserve_attachments: *const u32,
}

#[repr(C)]
struct VkRenderPassCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    attachment_count: u32,
    p_attachments: *const VkAttachmentDescription,
    subpass_count: u32,
    p_subpasses: *const VkSubpassDescription,
    dependency_count: u32,
    p_dependencies: *const c_void,
}

#[repr(C)]
struct VkFramebufferCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    render_pass: VkRenderPass,
    attachment_count: u32,
    p_attachments: *const VkImageView,
    width: u32,
    height: u32,
    layers: u32,
}

#[repr(C)]
struct VkRenderPassBeginInfo {
    s_type: u32,
    p_next: *const c_void,
    render_pass: VkRenderPass,
    framebuffer: VkFramebuffer,
    render_area: VkRect2D,
    clear_value_count: u32,
    p_clear_values: *const VkClearColorValue,
}

#[repr(C)]
struct VkImageSubresourceLayers {
    aspect_mask: u32,
    mip_level: u32,
    base_array_layer: u32,
    layer_count: u32,
}

#[repr(C)]
struct VkOffset3D {
    x: i32,
    y: i32,
    z: i32,
}

#[repr(C)]
struct VkBufferImageCopy {
    buffer_offset: u64,
    buffer_row_length: u32,
    buffer_image_height: u32,
    image_subresource: VkImageSubresourceLayers,
    image_offset: VkOffset3D,
    image_extent: VkExtent3D,
}

// ---------------------------------------------------------------------------
// Function pointer types.
// ---------------------------------------------------------------------------

/// A yet-untyped Vulkan command, as `vkGet*ProcAddr` returns it.
type PfnVoid = unsafe extern "C" fn();

type PfnGetInstanceProcAddr = unsafe extern "C" fn(VkInstance, *const c_char) -> Option<PfnVoid>;
type PfnCreateInstance =
    unsafe extern "C" fn(*const VkInstanceCreateInfo, *const c_void, *mut VkInstance) -> VkResult;
type PfnDestroyInstance = unsafe extern "C" fn(VkInstance, *const c_void);
type PfnEnumeratePhysicalDevices =
    unsafe extern "C" fn(VkInstance, *mut u32, *mut VkPhysicalDevice) -> VkResult;
type PfnGetPhysicalDeviceProperties2 =
    unsafe extern "C" fn(VkPhysicalDevice, *mut VkPhysicalDeviceProperties2);
type PfnEnumerateDeviceExtensionProperties = unsafe extern "C" fn(
    VkPhysicalDevice,
    *const c_char,
    *mut u32,
    *mut VkExtensionProperties,
) -> VkResult;
type PfnGetPhysicalDeviceQueueFamilyProperties =
    unsafe extern "C" fn(VkPhysicalDevice, *mut u32, *mut VkQueueFamilyProperties);
type PfnCreateDevice = unsafe extern "C" fn(
    VkPhysicalDevice,
    *const VkDeviceCreateInfo,
    *const c_void,
    *mut VkDevice,
) -> VkResult;
type PfnGetDeviceProcAddr = unsafe extern "C" fn(VkDevice, *const c_char) -> Option<PfnVoid>;
type PfnDestroyDevice = unsafe extern "C" fn(VkDevice, *const c_void);
type PfnGetDeviceQueue = unsafe extern "C" fn(VkDevice, u32, u32, *mut VkQueue);
type PfnGetPhysicalDeviceMemoryProperties =
    unsafe extern "C" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryProperties);
type PfnGetPhysicalDeviceFormatProperties2 =
    unsafe extern "C" fn(VkPhysicalDevice, u32, *mut VkFormatProperties2);
type PfnCreateImage = unsafe extern "C" fn(
    VkDevice,
    *const VkImageCreateInfo,
    *const c_void,
    *mut VkImage,
) -> VkResult;
type PfnDestroyImage = unsafe extern "C" fn(VkDevice, VkImage, *const c_void);
type PfnGetImageMemoryRequirements =
    unsafe extern "C" fn(VkDevice, VkImage, *mut VkMemoryRequirements);
type PfnAllocateMemory = unsafe extern "C" fn(
    VkDevice,
    *const VkMemoryAllocateInfo,
    *const c_void,
    *mut VkDeviceMemory,
) -> VkResult;
type PfnFreeMemory = unsafe extern "C" fn(VkDevice, VkDeviceMemory, *const c_void);
type PfnBindImageMemory = unsafe extern "C" fn(VkDevice, VkImage, VkDeviceMemory, u64) -> VkResult;
type PfnGetImageDrmFormatModifierPropertiesEXT =
    unsafe extern "C" fn(VkDevice, VkImage, *mut VkImageDrmFormatModifierPropertiesEXT) -> VkResult;
type PfnGetImageSubresourceLayout =
    unsafe extern "C" fn(VkDevice, VkImage, *const VkImageSubresource, *mut VkSubresourceLayout);
type PfnGetMemoryFdKHR =
    unsafe extern "C" fn(VkDevice, *const VkMemoryGetFdInfoKHR, *mut i32) -> VkResult;
type PfnCreateCommandPool = unsafe extern "C" fn(
    VkDevice,
    *const VkCommandPoolCreateInfo,
    *const c_void,
    *mut VkCommandPool,
) -> VkResult;
type PfnDestroyCommandPool = unsafe extern "C" fn(VkDevice, VkCommandPool, *const c_void);
type PfnAllocateCommandBuffers = unsafe extern "C" fn(
    VkDevice,
    *const VkCommandBufferAllocateInfo,
    *mut VkCommandBuffer,
) -> VkResult;
type PfnFreeCommandBuffers =
    unsafe extern "C" fn(VkDevice, VkCommandPool, u32, *const VkCommandBuffer);
type PfnBeginCommandBuffer =
    unsafe extern "C" fn(VkCommandBuffer, *const VkCommandBufferBeginInfo) -> VkResult;
type PfnEndCommandBuffer = unsafe extern "C" fn(VkCommandBuffer) -> VkResult;
type PfnCmdPipelineBarrier = unsafe extern "C" fn(
    VkCommandBuffer,
    u32,
    u32,
    u32,
    u32,
    *const c_void,
    u32,
    *const c_void,
    u32,
    *const VkImageMemoryBarrier,
);
type PfnQueueSubmit = unsafe extern "C" fn(VkQueue, u32, *const VkSubmitInfo, VkFence) -> VkResult;
type PfnCreateFence = unsafe extern "C" fn(
    VkDevice,
    *const VkFenceCreateInfo,
    *const c_void,
    *mut VkFence,
) -> VkResult;
type PfnDestroyFence = unsafe extern "C" fn(VkDevice, VkFence, *const c_void);
type PfnWaitForFences = unsafe extern "C" fn(VkDevice, u32, *const VkFence, u32, u64) -> VkResult;
type PfnResetFences = unsafe extern "C" fn(VkDevice, u32, *const VkFence) -> VkResult;
type PfnCreateSemaphore = unsafe extern "C" fn(
    VkDevice,
    *const VkSemaphoreCreateInfo,
    *const c_void,
    *mut VkSemaphore,
) -> VkResult;
type PfnDestroySemaphore = unsafe extern "C" fn(VkDevice, VkSemaphore, *const c_void);
type PfnGetSemaphoreFdKHR =
    unsafe extern "C" fn(VkDevice, *const VkSemaphoreGetFdInfoKHR, *mut i32) -> VkResult;
type PfnImportSemaphoreFdKHR =
    unsafe extern "C" fn(VkDevice, *const VkImportSemaphoreFdInfoKHR) -> VkResult;
type PfnDeviceWaitIdle = unsafe extern "C" fn(VkDevice) -> VkResult;
type PfnCreateBuffer = unsafe extern "C" fn(
    VkDevice,
    *const VkBufferCreateInfo,
    *const c_void,
    *mut VkBuffer,
) -> VkResult;
type PfnDestroyBuffer = unsafe extern "C" fn(VkDevice, VkBuffer, *const c_void);
type PfnGetBufferMemoryRequirements =
    unsafe extern "C" fn(VkDevice, VkBuffer, *mut VkMemoryRequirements);
type PfnBindBufferMemory =
    unsafe extern "C" fn(VkDevice, VkBuffer, VkDeviceMemory, u64) -> VkResult;
type PfnMapMemory =
    unsafe extern "C" fn(VkDevice, VkDeviceMemory, u64, u64, u32, *mut *mut c_void) -> VkResult;
type PfnCreateShaderModule = unsafe extern "C" fn(
    VkDevice,
    *const VkShaderModuleCreateInfo,
    *const c_void,
    *mut VkShaderModule,
) -> VkResult;
type PfnDestroyShaderModule = unsafe extern "C" fn(VkDevice, VkShaderModule, *const c_void);
type PfnCreatePipelineLayout = unsafe extern "C" fn(
    VkDevice,
    *const VkPipelineLayoutCreateInfo,
    *const c_void,
    *mut VkPipelineLayout,
) -> VkResult;
type PfnDestroyPipelineLayout = unsafe extern "C" fn(VkDevice, VkPipelineLayout, *const c_void);
type PfnCreateGraphicsPipelines = unsafe extern "C" fn(
    VkDevice,
    u64,
    u32,
    *const VkGraphicsPipelineCreateInfo,
    *const c_void,
    *mut VkPipeline,
) -> VkResult;
type PfnDestroyPipeline = unsafe extern "C" fn(VkDevice, VkPipeline, *const c_void);
type PfnCreateDescriptorSetLayout = unsafe extern "C" fn(
    VkDevice,
    *const VkDescriptorSetLayoutCreateInfo,
    *const c_void,
    *mut VkDescriptorSetLayout,
) -> VkResult;
type PfnDestroyDescriptorSetLayout =
    unsafe extern "C" fn(VkDevice, VkDescriptorSetLayout, *const c_void);
type PfnCreateDescriptorPool = unsafe extern "C" fn(
    VkDevice,
    *const VkDescriptorPoolCreateInfo,
    *const c_void,
    *mut VkDescriptorPool,
) -> VkResult;
type PfnDestroyDescriptorPool = unsafe extern "C" fn(VkDevice, VkDescriptorPool, *const c_void);
type PfnAllocateDescriptorSets = unsafe extern "C" fn(
    VkDevice,
    *const VkDescriptorSetAllocateInfo,
    *mut VkDescriptorSet,
) -> VkResult;
type PfnFreeDescriptorSets =
    unsafe extern "C" fn(VkDevice, VkDescriptorPool, u32, *const VkDescriptorSet) -> VkResult;
type PfnUpdateDescriptorSets =
    unsafe extern "C" fn(VkDevice, u32, *const VkWriteDescriptorSet, u32, *const c_void);
type PfnCreateSampler = unsafe extern "C" fn(
    VkDevice,
    *const VkSamplerCreateInfo,
    *const c_void,
    *mut VkSampler,
) -> VkResult;
type PfnDestroySampler = unsafe extern "C" fn(VkDevice, VkSampler, *const c_void);
type PfnCreateRenderPass = unsafe extern "C" fn(
    VkDevice,
    *const VkRenderPassCreateInfo,
    *const c_void,
    *mut VkRenderPass,
) -> VkResult;
type PfnDestroyRenderPass = unsafe extern "C" fn(VkDevice, VkRenderPass, *const c_void);
type PfnCreateFramebuffer = unsafe extern "C" fn(
    VkDevice,
    *const VkFramebufferCreateInfo,
    *const c_void,
    *mut VkFramebuffer,
) -> VkResult;
type PfnDestroyFramebuffer = unsafe extern "C" fn(VkDevice, VkFramebuffer, *const c_void);
type PfnCreateImageView = unsafe extern "C" fn(
    VkDevice,
    *const VkImageViewCreateInfo,
    *const c_void,
    *mut VkImageView,
) -> VkResult;
type PfnDestroyImageView = unsafe extern "C" fn(VkDevice, VkImageView, *const c_void);
type PfnCmdBeginRenderPass =
    unsafe extern "C" fn(VkCommandBuffer, *const VkRenderPassBeginInfo, u32);
type PfnCmdEndRenderPass = unsafe extern "C" fn(VkCommandBuffer);
type PfnCmdBindPipeline = unsafe extern "C" fn(VkCommandBuffer, u32, VkPipeline);
type PfnCmdBindVertexBuffers =
    unsafe extern "C" fn(VkCommandBuffer, u32, u32, *const VkBuffer, *const u64);
type PfnCmdBindDescriptorSets = unsafe extern "C" fn(
    VkCommandBuffer,
    u32,
    VkPipelineLayout,
    u32,
    u32,
    *const VkDescriptorSet,
    u32,
    *const u32,
);
type PfnCmdPushConstants =
    unsafe extern "C" fn(VkCommandBuffer, VkPipelineLayout, u32, u32, u32, *const c_void);
type PfnCmdSetViewport = unsafe extern "C" fn(VkCommandBuffer, u32, u32, *const VkViewport);
type PfnCmdSetScissor = unsafe extern "C" fn(VkCommandBuffer, u32, u32, *const VkRect2D);
type PfnCmdDraw = unsafe extern "C" fn(VkCommandBuffer, u32, u32, u32, u32);
type PfnCmdCopyBufferToImage =
    unsafe extern "C" fn(VkCommandBuffer, VkBuffer, VkImage, u32, u32, *const VkBufferImageCopy);
type PfnCmdCopyImageToBuffer =
    unsafe extern "C" fn(VkCommandBuffer, VkImage, u32, VkBuffer, u32, *const VkBufferImageCopy);

/// Fetch an instance-level command by name, as a specific pointer type.
///
/// # Safety
/// `T` must be the exact PFN type of `name` per the Vulkan spec.
unsafe fn load<T>(gipa: PfnGetInstanceProcAddr, instance: VkInstance, name: &CStr) -> Result<T> {
    // SAFETY: gipa is the loader's vkGetInstanceProcAddr; name is a command name.
    let p = unsafe { gipa(instance, name.as_ptr()) }
        .ok_or_else(|| Error::msg(format!("vulkan loader lacks {}", name.to_string_lossy())))?;
    debug_assert_eq!(size_of::<T>(), size_of::<PfnVoid>());
    // SAFETY: caller guarantees T is the command's true signature; both are
    // fn-pointer sized (asserted above).
    Ok(unsafe { core::mem::transmute_copy::<PfnVoid, T>(&p) })
}

/// Map a VkResult to our error type with the failing call named.
fn check(result: VkResult, what: &str) -> Result<()> {
    if result == VK_SUCCESS {
        Ok(())
    } else {
        Err(Error::msg(format!("{what} failed: VkResult {result}")))
    }
}

// ---------------------------------------------------------------------------
// Owned handles.
// ---------------------------------------------------------------------------

/// The instance and its destructor, so any error after creation still tears it
/// down on drop.
struct Instance {
    raw: VkInstance,
    destroy: PfnDestroyInstance,
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: raw came from vkCreateInstance and is destroyed exactly once,
        // after every object created from it (field order in Gpu).
        unsafe { (self.destroy)(self.raw, core::ptr::null()) };
    }
}

/// The logical device and its destructor.
struct Device {
    raw: VkDevice,
    destroy: PfnDestroyDevice,
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: raw came from vkCreateDevice and is destroyed exactly once.
        unsafe { (self.destroy)(self.raw, core::ptr::null()) };
    }
}

// ---------------------------------------------------------------------------
// Device function table.
// ---------------------------------------------------------------------------

/// Every device-level command the render path uses, loaded once at bring-up so
/// a driver missing one fails there instead of mid-frame.
struct DeviceFns {
    create_image: PfnCreateImage,
    destroy_image: PfnDestroyImage,
    get_image_memory_requirements: PfnGetImageMemoryRequirements,
    allocate_memory: PfnAllocateMemory,
    free_memory: PfnFreeMemory,
    bind_image_memory: PfnBindImageMemory,
    get_image_drm_format_modifier_properties: PfnGetImageDrmFormatModifierPropertiesEXT,
    get_image_subresource_layout: PfnGetImageSubresourceLayout,
    get_memory_fd: PfnGetMemoryFdKHR,
    create_command_pool: PfnCreateCommandPool,
    destroy_command_pool: PfnDestroyCommandPool,
    allocate_command_buffers: PfnAllocateCommandBuffers,
    free_command_buffers: PfnFreeCommandBuffers,
    begin_command_buffer: PfnBeginCommandBuffer,
    end_command_buffer: PfnEndCommandBuffer,
    cmd_pipeline_barrier: PfnCmdPipelineBarrier,
    queue_submit: PfnQueueSubmit,
    create_fence: PfnCreateFence,
    destroy_fence: PfnDestroyFence,
    wait_for_fences: PfnWaitForFences,
    reset_fences: PfnResetFences,
    create_semaphore: PfnCreateSemaphore,
    destroy_semaphore: PfnDestroySemaphore,
    get_semaphore_fd: PfnGetSemaphoreFdKHR,
    import_semaphore_fd: PfnImportSemaphoreFdKHR,
    device_wait_idle: PfnDeviceWaitIdle,
    create_buffer: PfnCreateBuffer,
    destroy_buffer: PfnDestroyBuffer,
    get_buffer_memory_requirements: PfnGetBufferMemoryRequirements,
    bind_buffer_memory: PfnBindBufferMemory,
    map_memory: PfnMapMemory,
    create_shader_module: PfnCreateShaderModule,
    destroy_shader_module: PfnDestroyShaderModule,
    create_pipeline_layout: PfnCreatePipelineLayout,
    destroy_pipeline_layout: PfnDestroyPipelineLayout,
    create_graphics_pipelines: PfnCreateGraphicsPipelines,
    destroy_pipeline: PfnDestroyPipeline,
    create_descriptor_set_layout: PfnCreateDescriptorSetLayout,
    destroy_descriptor_set_layout: PfnDestroyDescriptorSetLayout,
    create_descriptor_pool: PfnCreateDescriptorPool,
    destroy_descriptor_pool: PfnDestroyDescriptorPool,
    allocate_descriptor_sets: PfnAllocateDescriptorSets,
    free_descriptor_sets: PfnFreeDescriptorSets,
    update_descriptor_sets: PfnUpdateDescriptorSets,
    create_sampler: PfnCreateSampler,
    destroy_sampler: PfnDestroySampler,
    create_render_pass: PfnCreateRenderPass,
    destroy_render_pass: PfnDestroyRenderPass,
    create_framebuffer: PfnCreateFramebuffer,
    destroy_framebuffer: PfnDestroyFramebuffer,
    create_image_view: PfnCreateImageView,
    destroy_image_view: PfnDestroyImageView,
    cmd_begin_render_pass: PfnCmdBeginRenderPass,
    cmd_end_render_pass: PfnCmdEndRenderPass,
    cmd_bind_pipeline: PfnCmdBindPipeline,
    cmd_bind_vertex_buffers: PfnCmdBindVertexBuffers,
    cmd_bind_descriptor_sets: PfnCmdBindDescriptorSets,
    cmd_push_constants: PfnCmdPushConstants,
    cmd_set_viewport: PfnCmdSetViewport,
    cmd_set_scissor: PfnCmdSetScissor,
    cmd_draw: PfnCmdDraw,
    cmd_copy_buffer_to_image: PfnCmdCopyBufferToImage,
    cmd_copy_image_to_buffer: PfnCmdCopyImageToBuffer,
}

impl DeviceFns {
    fn load(gdpa: PfnGetDeviceProcAddr, device: VkDevice) -> Result<Self> {
        // SAFETY (each load): the name matches the transcribed PFN type.
        unsafe {
            Ok(Self {
                create_image: load_device(gdpa, device, c"vkCreateImage")?,
                destroy_image: load_device(gdpa, device, c"vkDestroyImage")?,
                get_image_memory_requirements: load_device(
                    gdpa,
                    device,
                    c"vkGetImageMemoryRequirements",
                )?,
                allocate_memory: load_device(gdpa, device, c"vkAllocateMemory")?,
                free_memory: load_device(gdpa, device, c"vkFreeMemory")?,
                bind_image_memory: load_device(gdpa, device, c"vkBindImageMemory")?,
                get_image_drm_format_modifier_properties: load_device(
                    gdpa,
                    device,
                    c"vkGetImageDrmFormatModifierPropertiesEXT",
                )?,
                get_image_subresource_layout: load_device(
                    gdpa,
                    device,
                    c"vkGetImageSubresourceLayout",
                )?,
                get_memory_fd: load_device(gdpa, device, c"vkGetMemoryFdKHR")?,
                create_command_pool: load_device(gdpa, device, c"vkCreateCommandPool")?,
                destroy_command_pool: load_device(gdpa, device, c"vkDestroyCommandPool")?,
                allocate_command_buffers: load_device(gdpa, device, c"vkAllocateCommandBuffers")?,
                free_command_buffers: load_device(gdpa, device, c"vkFreeCommandBuffers")?,
                begin_command_buffer: load_device(gdpa, device, c"vkBeginCommandBuffer")?,
                end_command_buffer: load_device(gdpa, device, c"vkEndCommandBuffer")?,
                cmd_pipeline_barrier: load_device(gdpa, device, c"vkCmdPipelineBarrier")?,
                queue_submit: load_device(gdpa, device, c"vkQueueSubmit")?,
                create_fence: load_device(gdpa, device, c"vkCreateFence")?,
                destroy_fence: load_device(gdpa, device, c"vkDestroyFence")?,
                wait_for_fences: load_device(gdpa, device, c"vkWaitForFences")?,
                reset_fences: load_device(gdpa, device, c"vkResetFences")?,
                create_semaphore: load_device(gdpa, device, c"vkCreateSemaphore")?,
                destroy_semaphore: load_device(gdpa, device, c"vkDestroySemaphore")?,
                get_semaphore_fd: load_device(gdpa, device, c"vkGetSemaphoreFdKHR")?,
                import_semaphore_fd: load_device(gdpa, device, c"vkImportSemaphoreFdKHR")?,
                device_wait_idle: load_device(gdpa, device, c"vkDeviceWaitIdle")?,
                create_buffer: load_device(gdpa, device, c"vkCreateBuffer")?,
                destroy_buffer: load_device(gdpa, device, c"vkDestroyBuffer")?,
                get_buffer_memory_requirements: load_device(
                    gdpa,
                    device,
                    c"vkGetBufferMemoryRequirements",
                )?,
                bind_buffer_memory: load_device(gdpa, device, c"vkBindBufferMemory")?,
                map_memory: load_device(gdpa, device, c"vkMapMemory")?,
                create_shader_module: load_device(gdpa, device, c"vkCreateShaderModule")?,
                destroy_shader_module: load_device(gdpa, device, c"vkDestroyShaderModule")?,
                create_pipeline_layout: load_device(gdpa, device, c"vkCreatePipelineLayout")?,
                destroy_pipeline_layout: load_device(gdpa, device, c"vkDestroyPipelineLayout")?,
                create_graphics_pipelines: load_device(gdpa, device, c"vkCreateGraphicsPipelines")?,
                destroy_pipeline: load_device(gdpa, device, c"vkDestroyPipeline")?,
                create_descriptor_set_layout: load_device(
                    gdpa,
                    device,
                    c"vkCreateDescriptorSetLayout",
                )?,
                destroy_descriptor_set_layout: load_device(
                    gdpa,
                    device,
                    c"vkDestroyDescriptorSetLayout",
                )?,
                create_descriptor_pool: load_device(gdpa, device, c"vkCreateDescriptorPool")?,
                destroy_descriptor_pool: load_device(gdpa, device, c"vkDestroyDescriptorPool")?,
                allocate_descriptor_sets: load_device(gdpa, device, c"vkAllocateDescriptorSets")?,
                free_descriptor_sets: load_device(gdpa, device, c"vkFreeDescriptorSets")?,
                update_descriptor_sets: load_device(gdpa, device, c"vkUpdateDescriptorSets")?,
                create_sampler: load_device(gdpa, device, c"vkCreateSampler")?,
                destroy_sampler: load_device(gdpa, device, c"vkDestroySampler")?,
                create_render_pass: load_device(gdpa, device, c"vkCreateRenderPass")?,
                destroy_render_pass: load_device(gdpa, device, c"vkDestroyRenderPass")?,
                create_framebuffer: load_device(gdpa, device, c"vkCreateFramebuffer")?,
                destroy_framebuffer: load_device(gdpa, device, c"vkDestroyFramebuffer")?,
                create_image_view: load_device(gdpa, device, c"vkCreateImageView")?,
                destroy_image_view: load_device(gdpa, device, c"vkDestroyImageView")?,
                cmd_begin_render_pass: load_device(gdpa, device, c"vkCmdBeginRenderPass")?,
                cmd_end_render_pass: load_device(gdpa, device, c"vkCmdEndRenderPass")?,
                cmd_bind_pipeline: load_device(gdpa, device, c"vkCmdBindPipeline")?,
                cmd_bind_vertex_buffers: load_device(gdpa, device, c"vkCmdBindVertexBuffers")?,
                cmd_bind_descriptor_sets: load_device(gdpa, device, c"vkCmdBindDescriptorSets")?,
                cmd_push_constants: load_device(gdpa, device, c"vkCmdPushConstants")?,
                cmd_set_viewport: load_device(gdpa, device, c"vkCmdSetViewport")?,
                cmd_set_scissor: load_device(gdpa, device, c"vkCmdSetScissor")?,
                cmd_draw: load_device(gdpa, device, c"vkCmdDraw")?,
                cmd_copy_buffer_to_image: load_device(gdpa, device, c"vkCmdCopyBufferToImage")?,
                cmd_copy_image_to_buffer: load_device(gdpa, device, c"vkCmdCopyImageToBuffer")?,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Bring-up.
// ---------------------------------------------------------------------------

/// A live Vulkan device matched to the compositor's main DRM device, with a
/// graphics queue, a command pool, and the format/memory facts the render
/// targets need. Every failure on the way here is a clean `Err` the caller
/// answers with the software path.
/// One sampled texture (an atlas, an image, or the 1x1 dummy) and whether its
/// contents have ever been defined (governs the upload barrier's old layout).
#[derive(Default)]
struct Texture {
    image: VkImage,
    memory: VkDeviceMemory,
    view: VkImageView,
    initialized: bool,
}

/// A host-visible, persistently mapped buffer (vertices or staging), grown by
/// recreation when a frame needs more.
#[derive(Default)]
struct HostBuffer {
    buf: VkBuffer,
    memory: VkDeviceMemory,
    ptr: *mut u8,
    capacity: usize,
}

/// The draw machinery shared by every frame: the one pipeline over the one
/// render pass, the descriptor plumbing, and the sampled textures (atlases,
/// per-image textures, and the dummy bound when a batch samples no image).
#[derive(Default)]
struct Renderer {
    render_pass: VkRenderPass,
    set_layout_atlas: VkDescriptorSetLayout,
    set_layout_image: VkDescriptorSetLayout,
    pipeline_layout: VkPipelineLayout,
    pipeline: VkPipeline,
    sampler: VkSampler,
    descriptor_pool: VkDescriptorPool,
    set_atlas: VkDescriptorSet,
    set_dummy: VkDescriptorSet,
    dummy: Texture,
    /// The R8 glyph atlas and BGRA emoji atlas textures, each tagged with the
    /// CPU mirror generation they were created for.
    glyph_tex: Option<(Texture, u32)>,
    emoji_tex: Option<(Texture, u32)>,
    /// One texture and descriptor set per distinct visible image, mirroring
    /// the CPU path's unbounded resample cache.
    images: std::collections::HashMap<ImageKey, (Texture, VkDescriptorSet)>,
}

pub struct Gpu {
    fns: DeviceFns,
    queue: VkQueue,
    queue_family: u32,
    pool: VkCommandPool,
    renderer: Renderer,
    /// The modifiers this device can render B8G8R8A8 into (single-plane,
    /// renderable, transfer-capable), to intersect with the compositor's list.
    modifiers: Vec<VkDrmFormatModifierPropertiesEXT>,
    /// Property flags per memory type index, for allocation.
    memory_type_flags: Vec<u32>,
    /// The chosen device's DRM render-node minor, for opening a fd to create
    /// explicit-sync sync objects. `None` if the device exposes no render node.
    render_node: Option<u32>,
    // Drop order is field order: the device before the instance, the instance
    // before the loader library. The underscored two are pure keep-alives.
    device: Device,
    _instance: Instance,
    _lib: DynLib,
    name: String,
}

impl Gpu {
    /// Bring up Vulkan against the DRM device `main_device` (a dev_t from
    /// dmabuf feedback), or against the first capable device when `None` (the
    /// offline pixel-parity harness, which has no compositor to match): load
    /// the loader, create an instance, pick the physical device, and create a
    /// one-queue logical device with the dmabuf presentation extensions.
    pub fn new(main_device: Option<u64>) -> Result<Self> {
        let lib = DynLib::open(c"libvulkan.so.1")?;
        let gipa_ptr = lib
            .sym(c"vkGetInstanceProcAddr")
            .ok_or_else(|| Error::msg("libvulkan.so.1 lacks vkGetInstanceProcAddr"))?;
        // SAFETY: the loader exports vkGetInstanceProcAddr with exactly this
        // signature; the pointer stays valid while `lib` lives.
        let gipa: PfnGetInstanceProcAddr = unsafe { core::mem::transmute(gipa_ptr) };

        let instance = create_instance(gipa)?;

        // SAFETY (all loads below): the names match the transcribed PFN types.
        let enumerate: PfnEnumeratePhysicalDevices =
            unsafe { load(gipa, instance.raw, c"vkEnumeratePhysicalDevices") }?;
        let get_props2: PfnGetPhysicalDeviceProperties2 =
            unsafe { load(gipa, instance.raw, c"vkGetPhysicalDeviceProperties2") }?;
        let enumerate_exts: PfnEnumerateDeviceExtensionProperties =
            unsafe { load(gipa, instance.raw, c"vkEnumerateDeviceExtensionProperties") }?;
        let get_queue_families: PfnGetPhysicalDeviceQueueFamilyProperties = unsafe {
            load(
                gipa,
                instance.raw,
                c"vkGetPhysicalDeviceQueueFamilyProperties",
            )
        }?;
        let create_device: PfnCreateDevice =
            unsafe { load(gipa, instance.raw, c"vkCreateDevice") }?;
        let gdpa: PfnGetDeviceProcAddr =
            unsafe { load(gipa, instance.raw, c"vkGetDeviceProcAddr") }?;
        let get_memory_props: PfnGetPhysicalDeviceMemoryProperties =
            unsafe { load(gipa, instance.raw, c"vkGetPhysicalDeviceMemoryProperties") }?;
        let get_format_props2: PfnGetPhysicalDeviceFormatProperties2 =
            unsafe { load(gipa, instance.raw, c"vkGetPhysicalDeviceFormatProperties2") }?;

        let physicals = enumerate_physicals(enumerate, instance.raw)?;
        let (physical, name, queue_family, render_node) = pick_device(
            get_props2,
            enumerate_exts,
            get_queue_families,
            &physicals,
            main_device,
        )?;

        let modifiers = render_modifiers(get_format_props2, physical);
        if modifiers.is_empty() {
            return Err(Error::msg(format!(
                "device {name} offers no usable DRM modifier for B8G8R8A8"
            )));
        }
        let memory_type_flags = memory_type_flags(get_memory_props, physical);

        let device = create_logical_device(create_device, gdpa, physical, queue_family)?;
        let fns = DeviceFns::load(gdpa, device.raw)?;

        let get_queue: PfnGetDeviceQueue =
            unsafe { load_device(gdpa, device.raw, c"vkGetDeviceQueue") }?;
        let mut queue: VkQueue = core::ptr::null_mut();
        // SAFETY: device is live and queue_family/index 0 were created with it.
        unsafe { get_queue(device.raw, queue_family, 0, &mut queue) };
        if queue.is_null() {
            return Err(Error::msg("vkGetDeviceQueue returned null"));
        }

        let pool_info = VkCommandPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queue_family_index: queue_family,
        };
        let mut pool: VkCommandPool = 0;
        // SAFETY: device is live; pool_info is fully initialized.
        check(
            unsafe {
                (fns.create_command_pool)(device.raw, &pool_info, core::ptr::null(), &mut pool)
            },
            "vkCreateCommandPool",
        )?;

        let renderer = match create_renderer(&fns, device.raw) {
            Ok(r) => r,
            Err(e) => {
                // SAFETY: pool is live and nothing was ever submitted.
                unsafe { (fns.destroy_command_pool)(device.raw, pool, core::ptr::null()) };
                return Err(e);
            }
        };

        Ok(Self {
            fns,
            queue,
            queue_family,
            pool,
            renderer,
            modifiers,
            memory_type_flags,
            render_node,
            device,
            _instance: instance,
            _lib: lib,
            name,
        })
    }

    /// The marketing name of the chosen device (for diagnostics).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The DRM render node minor of the chosen device (`/dev/dri/renderD<minor>`),
    /// where explicit sync opens a fd to create its sync objects. `None` when the
    /// device exposes no render node.
    pub fn render_node(&self) -> Option<u32> {
        self.render_node
    }

    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    /// The subset of the compositor's modifier list this device can render
    /// into, in the compositor's (preference) order. `DRM_FORMAT_MOD_INVALID`
    /// ("implicit layout") is dropped: explicit-modifier allocation is the
    /// whole point of this path.
    pub fn image_modifiers(&self, compositor: &[u64]) -> Vec<u64> {
        compositor
            .iter()
            .copied()
            .filter(|&m| m != DRM_FORMAT_MOD_INVALID)
            .filter(|&m| self.modifiers.iter().any(|p| p.drm_format_modifier == m))
            .collect()
    }

    /// Every modifier this device can render into, for the offline parity
    /// harness where there is no compositor list to intersect with.
    pub fn own_modifiers(&self) -> Vec<u64> {
        self.modifiers
            .iter()
            .map(|p| p.drm_format_modifier)
            .collect()
    }

    /// Whether an image texture for `key` is already uploaded (the batcher
    /// asks, and attaches bitmaps only for the ones this renderer lacks).
    pub fn has_image(&self, key: &ImageKey) -> bool {
        self.renderer.images.contains_key(key)
    }

    /// Block until the queue drains; teardown and resize use this so nothing
    /// is destroyed while a frame is in flight.
    pub fn wait_idle(&self) {
        // SAFETY: device is live. The result only matters to callers about to
        // destroy things, and a lost device makes destruction safe anyway.
        let _ = unsafe { (self.fns.device_wait_idle)(self.device.raw) };
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.wait_idle();
        let renderer = std::mem::take(&mut self.renderer);
        renderer.destroy(&self.fns, self.device.raw);
        // SAFETY: pool came from vkCreateCommandPool on this device; command
        // buffers still allocated from it are freed with it.
        unsafe { (self.fns.destroy_command_pool)(self.device.raw, self.pool, core::ptr::null()) };
    }
}

/// The device's single-plane, color-attachment-capable modifiers for
/// B8G8R8A8_UNORM (the two-call pattern over the chained modifier list).
fn render_modifiers(
    get_format_props2: PfnGetPhysicalDeviceFormatProperties2,
    physical: VkPhysicalDevice,
) -> Vec<VkDrmFormatModifierPropertiesEXT> {
    let mut list = VkDrmFormatModifierPropertiesListEXT {
        s_type: VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
        p_next: core::ptr::null_mut(),
        drm_format_modifier_count: 0,
        p_drm_format_modifier_properties: core::ptr::null_mut(),
    };
    let mut props = VkFormatProperties2 {
        s_type: VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
        p_next: (&mut list as *mut VkDrmFormatModifierPropertiesListEXT).cast(),
        format_properties: VkFormatProperties {
            linear_tiling_features: 0,
            optimal_tiling_features: 0,
            buffer_features: 0,
        },
    };
    // SAFETY: physical is live; props chains one correctly-typed list struct.
    unsafe { get_format_props2(physical, VK_FORMAT_B8G8R8A8_UNORM, &mut props) };
    let mut mods = vec![
        VkDrmFormatModifierPropertiesEXT {
            drm_format_modifier: 0,
            drm_format_modifier_plane_count: 0,
            drm_format_modifier_tiling_features: 0,
        };
        list.drm_format_modifier_count as usize
    ];
    list.p_drm_format_modifier_properties = mods.as_mut_ptr();
    // SAFETY: the array now has room for the advertised count.
    unsafe { get_format_props2(physical, VK_FORMAT_B8G8R8A8_UNORM, &mut props) };
    mods.truncate(list.drm_format_modifier_count as usize);
    let needed = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT
        | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
        | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    mods.retain(|m| {
        m.drm_format_modifier_plane_count == 1
            && m.drm_format_modifier_tiling_features & needed == needed
    });
    mods
}

/// Property flags per memory type index.
fn memory_type_flags(
    get_memory_props: PfnGetPhysicalDeviceMemoryProperties,
    physical: VkPhysicalDevice,
) -> Vec<u32> {
    // SAFETY: zeroed VkPhysicalDeviceMemoryProperties (integers) is a valid
    // out-parameter.
    let mut props = unsafe { core::mem::zeroed::<VkPhysicalDeviceMemoryProperties>() };
    // SAFETY: physical is live; props is sized per the ABI test.
    unsafe { get_memory_props(physical, &mut props) };
    props
        .memory_types
        .iter()
        .take((props.memory_type_count as usize).min(32))
        .map(|t| t.property_flags)
        .collect()
}

// ---------------------------------------------------------------------------
// Render targets.
// ---------------------------------------------------------------------------

/// One presentable render target: a device-local image whose memory is
/// exported as the dmabuf behind a `wl_buffer`, plus the per-image submission
/// state (command buffer, completion fence, and the two semaphores of the
/// implicit-sync bridge). Created and destroyed through [`Gpu`]; it holds raw
/// handles, so dropping it without [`Gpu::destroy_image`] leaks GPU memory
/// until the device goes down.
pub struct GpuImage {
    image: VkImage,
    memory: VkDeviceMemory,
    /// The exported dmabuf, kept for the sync-file ioctls; the compositor
    /// holds its own reference once the wl_buffer imports it.
    fd: OwnedFd,
    cmd: VkCommandBuffer,
    /// Signaled when the last submission touching this image completed.
    fence: VkFence,
    /// Carries the compositor's read fences into the next submission (a
    /// temporary sync-fd import per frame).
    acquire: VkSemaphore,
    /// Signaled by each submission; exported as a sync file and attached to
    /// the dmabuf so the compositor's reads wait for the render.
    release: VkSemaphore,
    /// The render-pass view of the image and its framebuffer.
    view: VkImageView,
    framebuffer: VkFramebuffer,
    /// This frame slot's vertex buffer and upload staging, safe to rewrite
    /// once the slot's fence has signaled.
    vertices: HostBuffer,
    staging: HostBuffer,
    width: u32,
    height: u32,
    stride: u32,
    offset: u32,
    modifier: u64,
    /// False until the first submission: the first frame enters from
    /// UNDEFINED, later frames re-acquire from the foreign queue family.
    used: bool,
}

impl GpuImage {
    /// The dmabuf to hand to `zwp_linux_buffer_params_v1.add`.
    pub fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    pub fn stride(&self) -> u32 {
        self.stride
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }

    /// The modifier the driver actually chose from the allowed list.
    pub fn modifier(&self) -> u64 {
        self.modifier
    }
}

/// Partially constructed image state, so an error mid-`create_image` destroys
/// exactly what exists so far (handle 0 = never created).
struct PendingImage<'a> {
    gpu: &'a Gpu,
    image: VkImage,
    memory: VkDeviceMemory,
    fence: VkFence,
    acquire: VkSemaphore,
    release: VkSemaphore,
    cmd: VkCommandBuffer,
    view: VkImageView,
    framebuffer: VkFramebuffer,
    vertices: HostBuffer,
    staging: HostBuffer,
}

impl Drop for PendingImage<'_> {
    fn drop(&mut self) {
        let d = self.gpu.device.raw;
        let f = &self.gpu.fns;
        // SAFETY: every non-zero handle below was created on this device and
        // is destroyed exactly once; nothing was ever submitted with them.
        unsafe {
            if self.framebuffer != 0 {
                (f.destroy_framebuffer)(d, self.framebuffer, core::ptr::null());
            }
            if self.view != 0 {
                (f.destroy_image_view)(d, self.view, core::ptr::null());
            }
        }
        self.gpu
            .destroy_host_buffer(std::mem::take(&mut self.vertices));
        self.gpu
            .destroy_host_buffer(std::mem::take(&mut self.staging));
        // SAFETY: as above; the fd (if exported) closes with its OwnedFd owner.
        unsafe {
            if !self.cmd.is_null() {
                (f.free_command_buffers)(d, self.gpu.pool, 1, &self.cmd);
            }
            if self.release != 0 {
                (f.destroy_semaphore)(d, self.release, core::ptr::null());
            }
            if self.acquire != 0 {
                (f.destroy_semaphore)(d, self.acquire, core::ptr::null());
            }
            if self.fence != 0 {
                (f.destroy_fence)(d, self.fence, core::ptr::null());
            }
            if self.memory != 0 {
                (f.free_memory)(d, self.memory, core::ptr::null());
            }
            if self.image != 0 {
                (f.destroy_image)(d, self.image, core::ptr::null());
            }
        }
    }
}

impl Gpu {
    /// Create a `width` x `height` render target, letting the driver pick a
    /// layout from `modifiers` (the compositor∩device intersection), and
    /// export its memory as a dmabuf. Returns the image with the geometry the
    /// wl_buffer needs (chosen modifier, plane stride and offset).
    pub fn create_image(&self, width: u32, height: u32, modifiers: &[u64]) -> Result<GpuImage> {
        if width == 0 || height == 0 {
            return Err(Error::msg("gpu image with a zero dimension"));
        }
        if modifiers.is_empty() {
            return Err(Error::msg("gpu image with no allowed modifiers"));
        }
        let d = self.device.raw;
        let f = &self.fns;
        let mut pending = PendingImage {
            gpu: self,
            image: 0,
            memory: 0,
            fence: 0,
            acquire: 0,
            release: 0,
            cmd: core::ptr::null_mut(),
            view: 0,
            framebuffer: 0,
            vertices: HostBuffer::default(),
            staging: HostBuffer::default(),
        };

        // The image: renderable, clearable, exportable, laid out by one of the
        // allowed modifiers (driver's choice, queried back below).
        let modifier_list = VkImageDrmFormatModifierListCreateInfoEXT {
            s_type: VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
            p_next: core::ptr::null(),
            drm_format_modifier_count: modifiers.len() as u32,
            p_drm_format_modifiers: modifiers.as_ptr(),
        };
        let external = VkExternalMemoryImageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            p_next: (&modifier_list as *const VkImageDrmFormatModifierListCreateInfoEXT).cast(),
            handle_types: VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT,
        };
        let image_info = VkImageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            p_next: (&external as *const VkExternalMemoryImageCreateInfo).cast(),
            flags: 0,
            image_type: VK_IMAGE_TYPE_2D,
            format: VK_FORMAT_B8G8R8A8_UNORM,
            extent: VkExtent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: VK_SAMPLE_COUNT_1_BIT,
            tiling: VK_IMAGE_TILING_DRM_FORMAT_MODIFIER,
            // Rendered into by the pipeline; read back by the parity harness.
            usage: VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: core::ptr::null(),
            initial_layout: VK_IMAGE_LAYOUT_UNDEFINED,
        };
        // SAFETY: d is live; image_info and its chain outlive the call.
        check(
            unsafe { (f.create_image)(d, &image_info, core::ptr::null(), &mut pending.image) },
            "vkCreateImage",
        )?;

        // Dedicated, exportable, device-local memory.
        let mut reqs = VkMemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        // SAFETY: image is live; reqs is a valid out-parameter.
        unsafe { (f.get_image_memory_requirements)(d, pending.image, &mut reqs) };
        let memory_type = self
            .pick_memory_type(reqs.memory_type_bits, 0)
            .ok_or_else(|| Error::msg("no memory type fits the exported image"))?;
        let dedicated = VkMemoryDedicatedAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            image: pending.image,
            buffer: 0,
        };
        let export = VkExportMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            p_next: (&dedicated as *const VkMemoryDedicatedAllocateInfo).cast(),
            handle_types: VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT,
        };
        let alloc = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: (&export as *const VkExportMemoryAllocateInfo).cast(),
            allocation_size: reqs.size,
            memory_type_index: memory_type,
        };
        // SAFETY: alloc and its chain outlive the call.
        check(
            unsafe { (f.allocate_memory)(d, &alloc, core::ptr::null(), &mut pending.memory) },
            "vkAllocateMemory",
        )?;
        // SAFETY: image and memory are live and compatible (requirements above).
        check(
            unsafe { (f.bind_image_memory)(d, pending.image, pending.memory, 0) },
            "vkBindImageMemory",
        )?;

        // What layout did the driver pick? The wl_buffer must describe it.
        let mut chosen = VkImageDrmFormatModifierPropertiesEXT {
            s_type: VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT,
            p_next: core::ptr::null_mut(),
            drm_format_modifier: 0,
        };
        // SAFETY: image is live; chosen is a valid out-parameter.
        check(
            unsafe { (f.get_image_drm_format_modifier_properties)(d, pending.image, &mut chosen) },
            "vkGetImageDrmFormatModifierPropertiesEXT",
        )?;
        let subresource = VkImageSubresource {
            aspect_mask: VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
            mip_level: 0,
            array_layer: 0,
        };
        let mut layout = VkSubresourceLayout {
            offset: 0,
            size: 0,
            row_pitch: 0,
            array_pitch: 0,
            depth_pitch: 0,
        };
        // SAFETY: image is live; the plane-0 aspect is valid for modifier tiling.
        unsafe { (f.get_image_subresource_layout)(d, pending.image, &subresource, &mut layout) };
        let stride = u32::try_from(layout.row_pitch)
            .map_err(|_| Error::msg("image row pitch exceeds the wl_buffer stride range"))?;
        let offset = u32::try_from(layout.offset)
            .map_err(|_| Error::msg("image plane offset exceeds the wl_buffer range"))?;

        // Export the dmabuf.
        let get_fd = VkMemoryGetFdInfoKHR {
            s_type: VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
            p_next: core::ptr::null(),
            memory: pending.memory,
            handle_type: VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT,
        };
        let mut raw_fd: i32 = -1;
        // SAFETY: memory is live; raw_fd receives a fresh fd we own.
        check(
            unsafe { (f.get_memory_fd)(d, &get_fd, &mut raw_fd) },
            "vkGetMemoryFdKHR",
        )?;
        if raw_fd < 0 {
            return Err(Error::msg("vkGetMemoryFdKHR returned an invalid fd"));
        }
        // SAFETY: the driver just handed us ownership of this fd.
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

        // Per-image submission state. The fence starts signaled so the first
        // frame's wait passes.
        let fence_info = VkFenceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0x1, // VK_FENCE_CREATE_SIGNALED_BIT
        };
        // SAFETY: fence_info outlives the call.
        check(
            unsafe { (f.create_fence)(d, &fence_info, core::ptr::null(), &mut pending.fence) },
            "vkCreateFence",
        )?;
        let plain_sem = VkSemaphoreCreateInfo {
            s_type: VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
        };
        // SAFETY: plain_sem outlives the call.
        check(
            unsafe { (f.create_semaphore)(d, &plain_sem, core::ptr::null(), &mut pending.acquire) },
            "vkCreateSemaphore (acquire)",
        )?;
        let export_sem = VkExportSemaphoreCreateInfo {
            s_type: VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
            p_next: core::ptr::null(),
            handle_types: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
        };
        let export_sem_info = VkSemaphoreCreateInfo {
            s_type: VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            p_next: (&export_sem as *const VkExportSemaphoreCreateInfo).cast(),
            flags: 0,
        };
        // SAFETY: export_sem_info and its chain outlive the call.
        check(
            unsafe {
                (f.create_semaphore)(d, &export_sem_info, core::ptr::null(), &mut pending.release)
            },
            "vkCreateSemaphore (release)",
        )?;

        let cmd_info = VkCommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            command_pool: self.pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        // SAFETY: the pool is live; one handle is written.
        check(
            unsafe { (f.allocate_command_buffers)(d, &cmd_info, &mut pending.cmd) },
            "vkAllocateCommandBuffers",
        )?;

        // The render-pass plumbing: a view of the image and its framebuffer.
        pending.view = self.create_view(pending.image, VK_FORMAT_B8G8R8A8_UNORM)?;
        let fb_info = VkFramebufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            render_pass: self.renderer.render_pass,
            attachment_count: 1,
            p_attachments: &pending.view,
            width,
            height,
            layers: 1,
        };
        // SAFETY: the render pass and view are live; fb_info outlives the call.
        check(
            unsafe {
                (f.create_framebuffer)(d, &fb_info, core::ptr::null(), &mut pending.framebuffer)
            },
            "vkCreateFramebuffer",
        )?;

        // Host-visible per-slot buffers; both grow on demand once a frame
        // needs more (the fence has been waited by then, so recreation is safe).
        pending.vertices = self.create_host_buffer(64 * 1024, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)?;
        pending.staging = self.create_host_buffer(
            256 * 1024,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        )?;

        // Success: move everything out of the guard.
        let img = GpuImage {
            image: pending.image,
            memory: pending.memory,
            fd,
            cmd: pending.cmd,
            fence: pending.fence,
            acquire: pending.acquire,
            release: pending.release,
            view: pending.view,
            framebuffer: pending.framebuffer,
            vertices: std::mem::take(&mut pending.vertices),
            staging: std::mem::take(&mut pending.staging),
            width,
            height,
            stride,
            offset,
            modifier: chosen.drm_format_modifier,
            used: false,
        };
        core::mem::forget(pending);
        Ok(img)
    }

    /// Tear an image down. The caller ensures no submission is in flight
    /// (waits for release + fence, or [`Gpu::wait_idle`]).
    pub fn destroy_image(&self, img: GpuImage) {
        let d = self.device.raw;
        let f = &self.fns;
        // SAFETY: all handles were created on this device by create_image and
        // are destroyed exactly once; the caller guarantees the GPU is done
        // with them. The dmabuf fd closes when img.fd drops.
        unsafe {
            (f.destroy_framebuffer)(d, img.framebuffer, core::ptr::null());
            (f.destroy_image_view)(d, img.view, core::ptr::null());
            (f.free_command_buffers)(d, self.pool, 1, &img.cmd);
            (f.destroy_semaphore)(d, img.release, core::ptr::null());
            (f.destroy_semaphore)(d, img.acquire, core::ptr::null());
            (f.destroy_fence)(d, img.fence, core::ptr::null());
        }
        self.destroy_host_buffer(img.vertices);
        self.destroy_host_buffer(img.staging);
        // SAFETY: as above.
        unsafe {
            (f.free_memory)(d, img.memory, core::ptr::null());
            (f.destroy_image)(d, img.image, core::ptr::null());
        }
    }

    /// Upload the frame's textures and vertices and record its command buffer,
    /// after waiting out the image's previous submission. The sync strategy
    /// (implicit bridge or explicit points) is layered on by the caller, which
    /// then submits; on return the command buffer is recorded but not yet in
    /// flight.
    fn prepare_frame(
        &mut self,
        img: &mut GpuImage,
        frame: &FrameData,
        background: [f32; 4],
    ) -> Result<()> {
        let d = self.device.raw;
        let f = &self.fns;

        // The image's previous submission must be complete before its command
        // buffer and buffers are rewritten (the wl_buffer release already
        // gates reuse of the *pixels*; this gates the CPU-visible state).
        // SAFETY: fence is live; the timeout bounds a broken driver.
        check(
            unsafe { (f.wait_for_fences)(d, 1, &img.fence, 1, FENCE_TIMEOUT_NS) },
            "vkWaitForFences",
        )?;
        // SAFETY: the fence is signaled and nothing else references it.
        check(
            unsafe { (f.reset_fences)(d, 1, &img.fence) },
            "vkResetFences",
        )?;

        // Textures first: (re)create atlases and image textures, assemble the
        // staging bytes and the copy list.
        let (staging_bytes, uploads) = self.sync_frame_textures(frame)?;
        if !staging_bytes.is_empty() {
            self.grow_host_buffer(
                &mut img.staging,
                staging_bytes.len(),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            )?;
            // SAFETY: staging is mapped with at least staging_bytes.len()
            // capacity (grown above) and the GPU is done with it (fence).
            unsafe {
                core::ptr::copy_nonoverlapping(
                    staging_bytes.as_ptr(),
                    img.staging.ptr,
                    staging_bytes.len(),
                );
            }
        }

        // Vertices.
        let vbytes = std::mem::size_of_val(frame.vertices.as_slice());
        if vbytes > 0 {
            self.grow_host_buffer(&mut img.vertices, vbytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)?;
            // SAFETY: the vertex buffer is mapped with capacity >= vbytes and
            // idle (fence); Vertex is repr(C) plain data.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    frame.vertices.as_ptr().cast::<u8>(),
                    img.vertices.ptr,
                    vbytes,
                );
            }
        }

        self.record_frame(img, frame, &uploads, background)?;
        Ok(())
    }

    /// Render `img` and present it under the implicit-sync bridge: acquire from the
    /// dmabuf's read fences, release the render fence back into the dmabuf (CPU wait
    /// if the bridge is unavailable). The fallback path for compositors that do not
    /// offer explicit sync. Returns with the frame in flight; the caller may
    /// attach/commit immediately.
    pub fn render_list(
        &mut self,
        img: &mut GpuImage,
        frame: &FrameData,
        background: [f32; 4],
    ) -> Result<()> {
        self.prepare_frame(img, frame, background)?;
        // Acquire: the compositor's unfinished reads become a wait semaphore.
        let wait_acquire = self.import_read_fences(img)?;
        self.submit_frame(img, wait_acquire)
    }

    /// Render `img` and present it under explicit sync: wait on `release_wait`
    /// (the compositor's release point for this buffer's previous frame, `None`
    /// on first use) instead of the dmabuf's implicit fences, and return the
    /// render-done sync file for the caller to publish as this frame's acquire
    /// point. `Ok(None)` means the driver could not export a fence, so the frame
    /// was CPU-waited to completion and the caller must set no points on the
    /// commit (the buffer is already safe to read).
    pub fn render_list_explicit(
        &mut self,
        img: &mut GpuImage,
        frame: &FrameData,
        background: [f32; 4],
        release_wait: Option<OwnedFd>,
    ) -> Result<Option<OwnedFd>> {
        self.prepare_frame(img, frame, background)?;
        let wait_acquire = self.import_acquire(img, release_wait)?;
        self.submit_frame_explicit(img, wait_acquire)
    }

    /// Read the image's pixels back (B,G,R,A byte order, tightly packed).
    /// Blocks until the copy completes; this is the pixel-parity harness's
    /// path, never the editor's.
    pub fn read_pixels(&mut self, img: &mut GpuImage) -> Result<Vec<u8>> {
        let d = self.device.raw;
        let f = &self.fns;
        let len = (img.width * img.height * 4) as usize;
        // SAFETY: fence is live; bounded wait.
        check(
            unsafe { (f.wait_for_fences)(d, 1, &img.fence, 1, FENCE_TIMEOUT_NS) },
            "vkWaitForFences",
        )?;
        // SAFETY: the fence is signaled.
        check(
            unsafe { (f.reset_fences)(d, 1, &img.fence) },
            "vkResetFences",
        )?;
        self.grow_host_buffer(
            &mut img.staging,
            len,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        )?;

        let begin = VkCommandBufferBeginInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: core::ptr::null(),
        };
        // SAFETY: the pool allows reset-on-begin; the fence was waited.
        check(
            unsafe { (f.begin_command_buffer)(img.cmd, &begin) },
            "vkBeginCommandBuffer",
        )?;
        let range = color_range();
        let (old_layout, src_qf, dst_qf) = self.acquire_params(img);
        // SAFETY (all cmd_* below): cmd is recording; structs outlive calls.
        unsafe {
            let acquire = VkImageMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: 0,
                dst_access_mask: VK_ACCESS_TRANSFER_READ_BIT,
                old_layout,
                new_layout: VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                src_queue_family_index: src_qf,
                dst_queue_family_index: dst_qf,
                image: img.image,
                subresource_range: range,
            };
            (f.cmd_pipeline_barrier)(
                img.cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &acquire,
            );
            let region = VkBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: VkImageSubresourceLayers {
                    aspect_mask: VK_IMAGE_ASPECT_COLOR_BIT,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: VkOffset3D { x: 0, y: 0, z: 0 },
                image_extent: VkExtent3D {
                    width: img.width,
                    height: img.height,
                    depth: 1,
                },
            };
            (f.cmd_copy_image_to_buffer)(
                img.cmd,
                img.image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                img.staging.buf,
                1,
                &region,
            );
            let release = VkImageMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: VK_ACCESS_TRANSFER_READ_BIT,
                dst_access_mask: 0,
                old_layout: VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                new_layout: VK_IMAGE_LAYOUT_GENERAL,
                src_queue_family_index: self.queue_family,
                dst_queue_family_index: VK_QUEUE_FAMILY_FOREIGN,
                image: img.image,
                subresource_range: range,
            };
            (f.cmd_pipeline_barrier)(
                img.cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &release,
            );
        }
        // SAFETY: cmd is recording.
        check(
            unsafe { (f.end_command_buffer)(img.cmd) },
            "vkEndCommandBuffer",
        )?;
        let submit = VkSubmitInfo {
            s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            p_next: core::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: core::ptr::null(),
            p_wait_dst_stage_mask: core::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &img.cmd,
            signal_semaphore_count: 0,
            p_signal_semaphores: core::ptr::null(),
        };
        // SAFETY: queue and cmd are live; submit outlives the call.
        check(
            unsafe { (f.queue_submit)(self.queue, 1, &submit, img.fence) },
            "vkQueueSubmit",
        )?;
        // SAFETY: fence is live; the wait leaves it signaled, matching the
        // signaled-at-rest protocol every entry point expects.
        check(
            unsafe { (f.wait_for_fences)(d, 1, &img.fence, 1, FENCE_TIMEOUT_NS) },
            "vkWaitForFences (readback)",
        )?;
        // SAFETY: staging is mapped, at least len bytes, and the GPU is done.
        let bytes = unsafe { core::slice::from_raw_parts(img.staging.ptr, len) }.to_vec();
        Ok(bytes)
    }

    /// The acquire side of the per-frame ownership dance: where the image is
    /// coming from (undefined on first use, the compositor after).
    fn acquire_params(&self, img: &GpuImage) -> (u32, u32, u32) {
        if img.used {
            (
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_FOREIGN,
                self.queue_family,
            )
        } else {
            (
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
            )
        }
    }

    /// Submit `img`'s recorded command buffer: wait on its acquire semaphore
    /// when `wait_acquire`, signal its release semaphore and fence. Sync out
    /// (bridge or explicit) is layered on by the callers below.
    fn submit_only(&self, img: &mut GpuImage, wait_acquire: bool) -> Result<()> {
        let f = &self.fns;
        let wait_stage: u32 = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        let submit = VkSubmitInfo {
            s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            p_next: core::ptr::null(),
            wait_semaphore_count: u32::from(wait_acquire),
            p_wait_semaphores: &img.acquire,
            p_wait_dst_stage_mask: &wait_stage,
            command_buffer_count: 1,
            p_command_buffers: &img.cmd,
            signal_semaphore_count: 1,
            p_signal_semaphores: &img.release,
        };
        // SAFETY: queue, cmd, semaphores, and fence are live; submit and its
        // pointees outlive the call.
        check(
            unsafe { (f.queue_submit)(self.queue, 1, &submit, img.fence) },
            "vkQueueSubmit",
        )?;
        img.used = true;
        Ok(())
    }

    /// Submit under the implicit-sync bridge: after the submit, the render-done
    /// fence rides the dmabuf so the compositor waits for it. If the bridge is
    /// unavailable, wait on the CPU instead; a slow correct frame beats a torn
    /// one.
    fn submit_frame(&self, img: &mut GpuImage, wait_acquire: bool) -> Result<()> {
        self.submit_only(img, wait_acquire)?;
        if !self.export_render_fence(img)? {
            // SAFETY: fence is live; the timeout bounds a broken driver.
            check(
                unsafe {
                    (self.fns.wait_for_fences)(self.device.raw, 1, &img.fence, 1, FENCE_TIMEOUT_NS)
                },
                "vkWaitForFences (bridge fallback)",
            )?;
        }
        Ok(())
    }

    /// Submit under explicit sync: after the submit, hand back the render-done
    /// sync file so the caller can publish it as this frame's acquire point. If
    /// the driver cannot export one, CPU-wait to completion and return `None` so
    /// the caller commits with no sync points.
    fn submit_frame_explicit(
        &self,
        img: &mut GpuImage,
        wait_acquire: bool,
    ) -> Result<Option<OwnedFd>> {
        self.submit_only(img, wait_acquire)?;
        match self.render_done_fence(img)? {
            Some(fd) => Ok(Some(fd)),
            None => {
                // SAFETY: fence is live; the timeout bounds a broken driver.
                check(
                    unsafe {
                        (self.fns.wait_for_fences)(
                            self.device.raw,
                            1,
                            &img.fence,
                            1,
                            FENCE_TIMEOUT_NS,
                        )
                    },
                    "vkWaitForFences (explicit fallback)",
                )?;
                Ok(None)
            }
        }
    }

    /// Import `sync` (a sync file) as a temporary payload of the acquire
    /// semaphore so the submission waits on it; `None` arms no wait. Returns
    /// whether a wait was armed. Consumes `sync` (Vulkan owns it on success).
    fn import_acquire(&self, img: &GpuImage, sync: Option<OwnedFd>) -> Result<bool> {
        let Some(sync) = sync else {
            return Ok(false);
        };
        let raw = sync.as_raw_fd();
        let info = VkImportSemaphoreFdInfoKHR {
            s_type: VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
            p_next: core::ptr::null(),
            semaphore: img.acquire,
            flags: VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
            handle_type: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
            fd: raw,
        };
        // SAFETY: info outlives the call; on success Vulkan owns the fd.
        let r = unsafe { (self.fns.import_semaphore_fd)(self.device.raw, &info) };
        if r == VK_SUCCESS {
            // Ownership moved to Vulkan; do not let OwnedFd close it.
            core::mem::forget(sync);
            Ok(true)
        } else {
            // Failed import: we keep ownership (sync drops); skip the wait.
            Ok(false)
        }
    }

    /// The implicit-sync bridge's acquire: the dmabuf's current read/write
    /// fences become the acquire wait.
    fn import_read_fences(&self, img: &GpuImage) -> Result<bool> {
        self.import_acquire(img, crate::ffi::dmabuf_export_sync_file(img.fd())?)
    }

    /// Export the just-submitted release semaphore as a sync file. `None` when
    /// the driver cannot, in which case the caller must CPU-wait.
    fn render_done_fence(&self, img: &GpuImage) -> Result<Option<OwnedFd>> {
        let info = VkSemaphoreGetFdInfoKHR {
            s_type: VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
            p_next: core::ptr::null(),
            semaphore: img.release,
            handle_type: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
        };
        let mut raw: i32 = -1;
        // SAFETY: the semaphore has a pending signal from the submit above,
        // which is what a sync-fd export requires.
        let r = unsafe { (self.fns.get_semaphore_fd)(self.device.raw, &info, &mut raw) };
        if r != VK_SUCCESS || raw < 0 {
            return Ok(None);
        }
        // SAFETY: the driver handed us ownership of this sync-file fd.
        Ok(Some(unsafe { OwnedFd::from_raw_fd(raw) }))
    }

    /// The implicit-sync bridge's release: attach the render-done fence to the
    /// dmabuf as its write fence so the compositor waits for it. Returns false
    /// when the kernel or driver cannot bridge.
    fn export_render_fence(&self, img: &GpuImage) -> Result<bool> {
        let Some(sync) = self.render_done_fence(img)? else {
            return Ok(false);
        };
        crate::ffi::dmabuf_import_sync_file(img.fd(), sync.as_raw_fd())
    }

    /// Record `img`'s command buffer for one frame: texture uploads, the
    /// acquire barrier, the render pass (clear plus batched quads), and the
    /// release back to the compositor in GENERAL layout.
    fn record_frame(
        &self,
        img: &GpuImage,
        frame: &FrameData,
        uploads: &[PendingUpload],
        background: [f32; 4],
    ) -> Result<()> {
        let f = &self.fns;
        let begin = VkCommandBufferBeginInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: core::ptr::null(),
        };
        // SAFETY: the pool allows reset-on-begin; the previous submission
        // completed (fence wait in render_list).
        check(
            unsafe { (f.begin_command_buffer)(img.cmd, &begin) },
            "vkBeginCommandBuffer",
        )?;

        for up in uploads {
            self.record_upload(img, up);
        }

        let range = color_range();
        let (old_layout, src_qf, dst_qf) = self.acquire_params(img);
        // SAFETY (all cmd_* to the end): cmd is in the recording state; every
        // struct outlives its call; handles are live.
        unsafe {
            let acquire = VkImageMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: 0,
                dst_access_mask: VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                old_layout,
                new_layout: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                src_queue_family_index: src_qf,
                dst_queue_family_index: dst_qf,
                image: img.image,
                subresource_range: range,
            };
            (f.cmd_pipeline_barrier)(
                img.cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &acquire,
            );

            let clear = VkClearColorValue {
                float32: background,
            };
            let pass = VkRenderPassBeginInfo {
                s_type: VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                p_next: core::ptr::null(),
                render_pass: self.renderer.render_pass,
                framebuffer: img.framebuffer,
                render_area: VkRect2D {
                    offset: VkOffset2D { x: 0, y: 0 },
                    extent: VkExtent2D {
                        width: img.width,
                        height: img.height,
                    },
                },
                clear_value_count: 1,
                p_clear_values: &clear,
            };
            (f.cmd_begin_render_pass)(img.cmd, &pass, VK_SUBPASS_CONTENTS_INLINE);

            if !frame.vertices.is_empty() {
                (f.cmd_bind_pipeline)(
                    img.cmd,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    self.renderer.pipeline,
                );
                let viewport = VkViewport {
                    x: 0.0,
                    y: 0.0,
                    width: img.width as f32,
                    height: img.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                (f.cmd_set_viewport)(img.cmd, 0, 1, &viewport);
                let scissor = VkRect2D {
                    offset: VkOffset2D { x: 0, y: 0 },
                    extent: VkExtent2D {
                        width: img.width,
                        height: img.height,
                    },
                };
                (f.cmd_set_scissor)(img.cmd, 0, 1, &scissor);
                let offset = 0u64;
                (f.cmd_bind_vertex_buffers)(img.cmd, 0, 1, &img.vertices.buf, &offset);
                let push: [f32; 2] = [img.width as f32, img.height as f32];
                (f.cmd_push_constants)(
                    img.cmd,
                    self.renderer.pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0,
                    8,
                    push.as_ptr().cast(),
                );
                (f.cmd_bind_descriptor_sets)(
                    img.cmd,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    self.renderer.pipeline_layout,
                    0,
                    1,
                    &self.renderer.set_atlas,
                    0,
                    core::ptr::null(),
                );
                // Track the bound image set so consecutive non-image batches
                // do not rebind the dummy.
                let mut bound: VkDescriptorSet = 0;
                for batch in &frame.batches {
                    let set = match &batch.image {
                        Some(key) => match self.renderer.images.get(key) {
                            Some((_, set)) => *set,
                            // The frame data always carries missing bitmaps,
                            // so this is unreachable; skipping the batch is
                            // the safe answer if it ever is not.
                            None => continue,
                        },
                        None => self.renderer.set_dummy,
                    };
                    if set != bound {
                        (f.cmd_bind_descriptor_sets)(
                            img.cmd,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            self.renderer.pipeline_layout,
                            1,
                            1,
                            &set,
                            0,
                            core::ptr::null(),
                        );
                        bound = set;
                    }
                    (f.cmd_draw)(img.cmd, batch.count, 1, batch.start, 0);
                }
            }

            (f.cmd_end_render_pass)(img.cmd);

            let release = VkImageMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dst_access_mask: 0,
                old_layout: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                new_layout: VK_IMAGE_LAYOUT_GENERAL,
                src_queue_family_index: self.queue_family,
                dst_queue_family_index: VK_QUEUE_FAMILY_FOREIGN,
                image: img.image,
                subresource_range: range,
            };
            (f.cmd_pipeline_barrier)(
                img.cmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &release,
            );
        }
        // SAFETY: cmd is in the recording state.
        check(
            unsafe { (f.end_command_buffer)(img.cmd) },
            "vkEndCommandBuffer",
        )
    }

    /// Record one texture upload (or a bare initialization barrier when the
    /// upload carries no bytes, the dummy texture's case).
    fn record_upload(&self, img: &GpuImage, up: &PendingUpload) {
        let f = &self.fns;
        let range = color_range();
        let (old_layout, src_stage, src_access) = if up.was_initialized {
            (
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT,
            )
        } else {
            (
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                0,
            )
        };
        // SAFETY (whole fn): cmd is recording; structs outlive their calls.
        unsafe {
            let to_dst = VkImageMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: src_access,
                dst_access_mask: VK_ACCESS_TRANSFER_WRITE_BIT,
                old_layout,
                new_layout: VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                image: up.image,
                subresource_range: range,
            };
            (f.cmd_pipeline_barrier)(
                img.cmd,
                src_stage,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &to_dst,
            );
            if up.w > 0 && up.h > 0 {
                let region = VkBufferImageCopy {
                    buffer_offset: up.staging_offset as u64,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: VkImageSubresourceLayers {
                        aspect_mask: VK_IMAGE_ASPECT_COLOR_BIT,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: VkOffset3D {
                        x: up.x as i32,
                        y: up.y as i32,
                        z: 0,
                    },
                    image_extent: VkExtent3D {
                        width: up.w,
                        height: up.h,
                        depth: 1,
                    },
                };
                (f.cmd_copy_buffer_to_image)(
                    img.cmd,
                    img.staging.buf,
                    up.image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    &region,
                );
            }
            let to_read = VkImageMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: VK_ACCESS_TRANSFER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_SHADER_READ_BIT,
                old_layout: VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                new_layout: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                image: up.image,
                subresource_range: range,
            };
            (f.cmd_pipeline_barrier)(
                img.cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &to_read,
            );
        }
    }

    /// Reconcile the renderer's textures with this frame's data: (re)create
    /// atlas textures on generation changes, create textures and descriptor
    /// sets for new images, and assemble the staging bytes plus copy list.
    fn sync_frame_textures(&mut self, frame: &FrameData) -> Result<(Vec<u8>, Vec<PendingUpload>)> {
        let mut staging = Vec::new();
        let mut uploads = Vec::new();

        // The dummy texture: 1x1, never written, bound whenever a batch has
        // no image. Contents are irrelevant (no mode samples it), only the
        // layout must be right, so its "upload" is a bare barrier.
        if self.renderer.dummy.image == 0 {
            let dummy = self.create_texture(VK_FORMAT_B8G8R8A8_UNORM, 1, 1)?;
            write_image_set(
                &self.fns,
                self.device.raw,
                self.renderer.set_dummy,
                0,
                self.renderer.sampler,
                dummy.view,
            );
            write_image_set(
                &self.fns,
                self.device.raw,
                self.renderer.set_dummy,
                0,
                self.renderer.sampler,
                dummy.view,
            );
            uploads.push(PendingUpload {
                image: dummy.image,
                was_initialized: false,
                staging_offset: 0,
                x: 0,
                y: 0,
                w: 0,
                h: 0,
            });
            self.renderer.dummy = dummy;
        }

        // Atlas textures follow the CPU mirrors' generations. A replacement
        // may still be sampled by the other slot's in-flight frame, so drain
        // the queue first (growth is rare).
        let glyph_gen = frame.glyph_atlas.generation;
        if self
            .renderer
            .glyph_tex
            .as_ref()
            .is_none_or(|(_, g)| *g != glyph_gen)
        {
            if let Some((old, _)) = self.renderer.glyph_tex.take() {
                self.wait_idle();
                destroy_texture(&self.fns, self.device.raw, old);
            }
            let tex = self.create_texture(
                VK_FORMAT_R8_UNORM,
                frame.glyph_atlas.width,
                frame.glyph_atlas.height,
            )?;
            self.renderer.glyph_tex = Some((tex, glyph_gen));
            self.update_atlas_set()?;
        }
        let emoji_gen = frame.emoji_atlas.generation;
        if self
            .renderer
            .emoji_tex
            .as_ref()
            .is_none_or(|(_, g)| *g != emoji_gen)
        {
            if let Some((old, _)) = self.renderer.emoji_tex.take() {
                self.wait_idle();
                destroy_texture(&self.fns, self.device.raw, old);
            }
            let tex = self.create_texture(
                VK_FORMAT_B8G8R8A8_UNORM,
                frame.emoji_atlas.width,
                frame.emoji_atlas.height,
            )?;
            self.renderer.emoji_tex = Some((tex, emoji_gen));
            self.update_atlas_set()?;
        }

        // Dirty-region atlas uploads.
        if let Some(up) = &frame.glyph_upload {
            if let Some((tex, _)) = &mut self.renderer.glyph_tex {
                uploads.push(PendingUpload {
                    image: tex.image,
                    was_initialized: tex.initialized,
                    staging_offset: staging.len(),
                    x: up.x,
                    y: up.y,
                    w: up.w,
                    h: up.h,
                });
                staging.extend_from_slice(&up.bytes);
                tex.initialized = true;
            }
        }
        if let Some(up) = &frame.emoji_upload {
            if let Some((tex, _)) = &mut self.renderer.emoji_tex {
                uploads.push(PendingUpload {
                    image: tex.image,
                    was_initialized: tex.initialized,
                    staging_offset: staging.len(),
                    x: up.x,
                    y: up.y,
                    w: up.w,
                    h: up.h,
                });
                staging.extend_from_slice(&up.bytes);
                tex.initialized = true;
            }
        }

        // New image textures, one per (path, display size).
        for (key, argb) in &frame.image_uploads {
            if self.renderer.images.contains_key(key) {
                continue;
            }
            let (w, h) = (key.1, key.2);
            let tex = self.create_texture(VK_FORMAT_B8G8R8A8_UNORM, w, h)?;
            let set = self.allocate_image_set(tex.view)?;
            uploads.push(PendingUpload {
                image: tex.image,
                was_initialized: false,
                staging_offset: staging.len(),
                x: 0,
                y: 0,
                w,
                h,
            });
            // ARGB words are B,G,R,A bytes in memory: B8G8R8A8 verbatim.
            for px in argb {
                staging.extend_from_slice(&px.to_ne_bytes());
            }
            let mut stored = tex;
            stored.initialized = true;
            self.renderer.images.insert(key.clone(), (stored, set));
        }

        // Bound the image-texture cache: resizes mint new (path, size) keys,
        // and the descriptor pool is finite. On overflow, evict everything
        // this frame does not draw (their sets return to the pool). The drain
        // makes eviction safe against the other slot's in-flight frame.
        const IMAGE_TEXTURE_CAP: usize = 64;
        if self.renderer.images.len() > IMAGE_TEXTURE_CAP {
            let used: std::collections::HashSet<&ImageKey> = frame
                .batches
                .iter()
                .filter_map(|b| b.image.as_ref())
                .collect();
            let evict: Vec<ImageKey> = self
                .renderer
                .images
                .keys()
                .filter(|k| !used.contains(k))
                .cloned()
                .collect();
            if !evict.is_empty() {
                self.wait_idle();
                for key in evict {
                    if let Some((tex, set)) = self.renderer.images.remove(&key) {
                        // SAFETY: the pool was created with the free flag and
                        // the set is no longer referenced (queue drained).
                        unsafe {
                            (self.fns.free_descriptor_sets)(
                                self.device.raw,
                                self.renderer.descriptor_pool,
                                1,
                                &set,
                            )
                        };
                        destroy_texture(&self.fns, self.device.raw, tex);
                    }
                }
            }
        }

        // Uploads whose staging_offset was pushed before later bytes landed
        // stay valid: offsets only ever append.
        Ok((staging, uploads))
    }

    /// Point the shared atlas descriptor set at the current atlas textures
    /// (both bindings, using the dummy for one that does not exist yet).
    fn update_atlas_set(&self) -> Result<()> {
        let glyph_view = self
            .renderer
            .glyph_tex
            .as_ref()
            .map(|(t, _)| t.view)
            .unwrap_or(self.renderer.dummy.view);
        let emoji_view = self
            .renderer
            .emoji_tex
            .as_ref()
            .map(|(t, _)| t.view)
            .unwrap_or(self.renderer.dummy.view);
        if glyph_view == 0 || emoji_view == 0 {
            return Err(Error::msg("atlas descriptor update before any texture"));
        }
        write_image_set(
            &self.fns,
            self.device.raw,
            self.renderer.set_atlas,
            0,
            self.renderer.sampler,
            glyph_view,
        );
        write_image_set(
            &self.fns,
            self.device.raw,
            self.renderer.set_atlas,
            1,
            self.renderer.sampler,
            emoji_view,
        );
        Ok(())
    }

    /// Allocate and fill a descriptor set for one image texture.
    fn allocate_image_set(&self, view: VkImageView) -> Result<VkDescriptorSet> {
        let info = VkDescriptorSetAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            descriptor_pool: self.renderer.descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &self.renderer.set_layout_image,
        };
        let mut set: VkDescriptorSet = 0;
        // SAFETY: the pool and layout are live; one handle is written.
        check(
            unsafe { (self.fns.allocate_descriptor_sets)(self.device.raw, &info, &mut set) },
            "vkAllocateDescriptorSets",
        )?;
        write_image_set(
            &self.fns,
            self.device.raw,
            set,
            0,
            self.renderer.sampler,
            view,
        );
        Ok(set)
    }

    /// A sampled, transfer-destination 2D texture (an atlas or an image).
    fn create_texture(&self, format: u32, width: u32, height: u32) -> Result<Texture> {
        let d = self.device.raw;
        let f = &self.fns;
        let info = VkImageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            image_type: VK_IMAGE_TYPE_2D,
            format,
            extent: VkExtent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: VK_SAMPLE_COUNT_1_BIT,
            tiling: VK_IMAGE_TILING_OPTIMAL,
            usage: VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: core::ptr::null(),
            initial_layout: VK_IMAGE_LAYOUT_UNDEFINED,
        };
        let mut image: VkImage = 0;
        // SAFETY: info outlives the call.
        check(
            unsafe { (f.create_image)(d, &info, core::ptr::null(), &mut image) },
            "vkCreateImage (texture)",
        )?;
        let mut reqs = VkMemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        // SAFETY: image is live.
        unsafe { (f.get_image_memory_requirements)(d, image, &mut reqs) };
        let Some(memory_type) = self.pick_memory_type(reqs.memory_type_bits, 0) else {
            // SAFETY: image is live and unbound.
            unsafe { (f.destroy_image)(d, image, core::ptr::null()) };
            return Err(Error::msg("no memory type fits a texture"));
        };
        let alloc = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            allocation_size: reqs.size,
            memory_type_index: memory_type,
        };
        let mut memory: VkDeviceMemory = 0;
        // SAFETY: alloc outlives the call.
        let r = unsafe { (f.allocate_memory)(d, &alloc, core::ptr::null(), &mut memory) };
        if r != VK_SUCCESS {
            // SAFETY: image is live and unbound.
            unsafe { (f.destroy_image)(d, image, core::ptr::null()) };
            return Err(Error::msg(format!(
                "vkAllocateMemory (texture) failed: VkResult {r}"
            )));
        }
        // SAFETY: image and memory are live and compatible.
        let r = unsafe { (f.bind_image_memory)(d, image, memory, 0) };
        if r != VK_SUCCESS {
            // SAFETY: both are live; the failed bind leaves them separable.
            unsafe {
                (f.free_memory)(d, memory, core::ptr::null());
                (f.destroy_image)(d, image, core::ptr::null());
            }
            return Err(Error::msg(format!(
                "vkBindImageMemory (texture) failed: VkResult {r}"
            )));
        }
        match self.create_view(image, format) {
            Ok(view) => Ok(Texture {
                image,
                memory,
                view,
                initialized: false,
            }),
            Err(e) => {
                // SAFETY: both are live; nothing else references them.
                unsafe {
                    (f.free_memory)(d, memory, core::ptr::null());
                    (f.destroy_image)(d, image, core::ptr::null());
                }
                Err(e)
            }
        }
    }

    /// A plain 2D color view of `image`.
    fn create_view(&self, image: VkImage, format: u32) -> Result<VkImageView> {
        let info = VkImageViewCreateInfo {
            s_type: VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            image,
            view_type: VK_IMAGE_VIEW_TYPE_2D,
            format,
            components: VkComponentMapping {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            },
            subresource_range: color_range(),
        };
        let mut view: VkImageView = 0;
        // SAFETY: image is live; info outlives the call.
        check(
            unsafe {
                (self.fns.create_image_view)(self.device.raw, &info, core::ptr::null(), &mut view)
            },
            "vkCreateImageView",
        )?;
        Ok(view)
    }

    /// A host-visible, host-coherent, persistently mapped buffer.
    fn create_host_buffer(&self, size: usize, usage: u32) -> Result<HostBuffer> {
        let d = self.device.raw;
        let f = &self.fns;
        let info = VkBufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            size: size as u64,
            usage,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: core::ptr::null(),
        };
        let mut buf: VkBuffer = 0;
        // SAFETY: info outlives the call.
        check(
            unsafe { (f.create_buffer)(d, &info, core::ptr::null(), &mut buf) },
            "vkCreateBuffer",
        )?;
        let mut reqs = VkMemoryRequirements {
            size: 0,
            alignment: 0,
            memory_type_bits: 0,
        };
        // SAFETY: buf is live.
        unsafe { (f.get_buffer_memory_requirements)(d, buf, &mut reqs) };
        let required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        let Some(memory_type) = self.pick_memory_type(reqs.memory_type_bits, required) else {
            // SAFETY: buf is live and unbound.
            unsafe { (f.destroy_buffer)(d, buf, core::ptr::null()) };
            return Err(Error::msg("no host-visible memory type for a buffer"));
        };
        let alloc = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            allocation_size: reqs.size,
            memory_type_index: memory_type,
        };
        let mut memory: VkDeviceMemory = 0;
        // SAFETY: alloc outlives the call.
        let r = unsafe { (f.allocate_memory)(d, &alloc, core::ptr::null(), &mut memory) };
        if r != VK_SUCCESS {
            // SAFETY: buf is live and unbound.
            unsafe { (f.destroy_buffer)(d, buf, core::ptr::null()) };
            return Err(Error::msg(format!(
                "vkAllocateMemory (buffer) failed: VkResult {r}"
            )));
        }
        // SAFETY: buf and memory are live and compatible.
        let r = unsafe { (f.bind_buffer_memory)(d, buf, memory, 0) };
        let mut ptr: *mut c_void = core::ptr::null_mut();
        // SAFETY: memory is live, host-visible, and bound (checked below).
        let r2 = if r == VK_SUCCESS {
            unsafe { (f.map_memory)(d, memory, 0, u64::MAX, 0, &mut ptr) }
        } else {
            r
        };
        if r2 != VK_SUCCESS || ptr.is_null() {
            // SAFETY: both are live; nothing else references them.
            unsafe {
                (f.free_memory)(d, memory, core::ptr::null());
                (f.destroy_buffer)(d, buf, core::ptr::null());
            }
            return Err(Error::msg(format!("buffer bind/map failed: VkResult {r2}")));
        }
        Ok(HostBuffer {
            buf,
            memory,
            ptr: ptr.cast(),
            capacity: size,
        })
    }

    /// Free a host buffer (a zeroed default is a no-op).
    fn destroy_host_buffer(&self, b: HostBuffer) {
        if b.buf == 0 {
            return;
        }
        // SAFETY: buf/memory are live and no submission references them (the
        // caller waited the slot's fence or the queue). Freeing unmaps.
        unsafe {
            (self.fns.destroy_buffer)(self.device.raw, b.buf, core::ptr::null());
            (self.fns.free_memory)(self.device.raw, b.memory, core::ptr::null());
        }
    }

    /// Grow `b` to at least `needed` bytes (doubling), preserving nothing:
    /// callers rewrite the whole buffer each frame.
    fn grow_host_buffer(&self, b: &mut HostBuffer, needed: usize, usage: u32) -> Result<()> {
        if needed <= b.capacity {
            return Ok(());
        }
        let new_cap = needed.max(b.capacity * 2);
        let fresh = self.create_host_buffer(new_cap, usage)?;
        let old = std::mem::replace(b, fresh);
        self.destroy_host_buffer(old);
        Ok(())
    }

    /// The first allowed memory type carrying every `required` property flag;
    /// with no requirement, prefer device-local and fall back to any allowed.
    fn pick_memory_type(&self, allowed_bits: u32, required: u32) -> Option<u32> {
        let allowed = |i: usize| allowed_bits & (1u32 << i) != 0;
        if required != 0 {
            return self
                .memory_type_flags
                .iter()
                .enumerate()
                .position(|(i, &flags)| allowed(i) && flags & required == required)
                .map(|i| i as u32);
        }
        self.memory_type_flags
            .iter()
            .enumerate()
            .position(|(i, &flags)| allowed(i) && flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0)
            .or_else(|| (0..self.memory_type_flags.len()).find(|&i| allowed(i)))
            .map(|i| i as u32)
    }
}

/// One texture copy scheduled for this frame's command buffer: which image,
/// where its bytes sit in the staging buffer, and the destination region.
/// A zero-sized region is a bare layout-initialization barrier.
struct PendingUpload {
    image: VkImage,
    was_initialized: bool,
    staging_offset: usize,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// The full-image color subresource range every barrier and view here uses.
fn color_range() -> VkImageSubresourceRange {
    VkImageSubresourceRange {
        aspect_mask: VK_IMAGE_ASPECT_COLOR_BIT,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Point `binding` of `set` at `view` through `sampler`.
fn write_image_set(
    fns: &DeviceFns,
    device: VkDevice,
    set: VkDescriptorSet,
    binding: u32,
    sampler: VkSampler,
    view: VkImageView,
) {
    let info = VkDescriptorImageInfo {
        sampler,
        image_view: view,
        image_layout: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    let write = VkWriteDescriptorSet {
        s_type: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        p_next: core::ptr::null(),
        dst_set: set,
        dst_binding: binding,
        dst_array_element: 0,
        descriptor_count: 1,
        descriptor_type: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        p_image_info: &info,
        p_buffer_info: core::ptr::null(),
        p_texel_buffer_view: core::ptr::null(),
    };
    // SAFETY: set, sampler, and view are live; write outlives the call.
    unsafe { (fns.update_descriptor_sets)(device, 1, &write, 0, core::ptr::null()) };
}

/// Destroy a texture's view, memory, and image (a zeroed default is a no-op).
fn destroy_texture(fns: &DeviceFns, device: VkDevice, t: Texture) {
    if t.image == 0 {
        return;
    }
    // SAFETY: all three were created together on this device and nothing in
    // flight references them (callers drain the queue where needed).
    unsafe {
        (fns.destroy_image_view)(device, t.view, core::ptr::null());
        (fns.free_memory)(device, t.memory, core::ptr::null());
        (fns.destroy_image)(device, t.image, core::ptr::null());
    }
}

impl Renderer {
    /// Destroy everything this renderer owns, in reverse creation order.
    /// Zeroed handles (a partial build) are skipped. The caller drained the
    /// queue.
    fn destroy(self, fns: &DeviceFns, device: VkDevice) {
        for (_, (tex, _)) in self.images {
            destroy_texture(fns, device, tex);
        }
        if let Some((tex, _)) = self.glyph_tex {
            destroy_texture(fns, device, tex);
        }
        if let Some((tex, _)) = self.emoji_tex {
            destroy_texture(fns, device, tex);
        }
        destroy_texture(fns, device, self.dummy);
        // SAFETY: every non-zero handle below came from this device and is
        // destroyed exactly once; destroying the pool frees its sets.
        unsafe {
            if self.descriptor_pool != 0 {
                (fns.destroy_descriptor_pool)(device, self.descriptor_pool, core::ptr::null());
            }
            if self.pipeline != 0 {
                (fns.destroy_pipeline)(device, self.pipeline, core::ptr::null());
            }
            if self.pipeline_layout != 0 {
                (fns.destroy_pipeline_layout)(device, self.pipeline_layout, core::ptr::null());
            }
            if self.sampler != 0 {
                (fns.destroy_sampler)(device, self.sampler, core::ptr::null());
            }
            if self.render_pass != 0 {
                (fns.destroy_render_pass)(device, self.render_pass, core::ptr::null());
            }
            if self.set_layout_atlas != 0 {
                (fns.destroy_descriptor_set_layout)(
                    device,
                    self.set_layout_atlas,
                    core::ptr::null(),
                );
            }
            if self.set_layout_image != 0 {
                (fns.destroy_descriptor_set_layout)(
                    device,
                    self.set_layout_image,
                    core::ptr::null(),
                );
            }
        }
    }
}

/// SPIR-V bytes (the committed blobs) as the u32 words Vulkan wants.
fn shader_words(bytes: &[u8]) -> Result<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) || bytes.is_empty() {
        return Err(Error::msg("malformed SPIR-V blob"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Build the frame renderer: the render pass, descriptor layouts, pipeline
/// (from the committed SPIR-V), sampler, and descriptor pool with the two
/// standing sets. On error, everything built so far is destroyed.
fn create_renderer(fns: &DeviceFns, device: VkDevice) -> Result<Renderer> {
    let mut r = Renderer::default();
    match fill_renderer(&mut r, fns, device) {
        Ok(()) => Ok(r),
        Err(e) => {
            r.destroy(fns, device);
            Err(e)
        }
    }
}

fn fill_renderer(r: &mut Renderer, fns: &DeviceFns, device: VkDevice) -> Result<()> {
    // The render pass: one B8G8R8A8 color attachment, cleared on load, kept
    // in COLOR_ATTACHMENT_OPTIMAL (the ownership barriers live outside).
    let attachment = VkAttachmentDescription {
        flags: 0,
        format: VK_FORMAT_B8G8R8A8_UNORM,
        samples: VK_SAMPLE_COUNT_1_BIT,
        load_op: VK_ATTACHMENT_LOAD_OP_CLEAR,
        store_op: VK_ATTACHMENT_STORE_OP_STORE,
        stencil_load_op: VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        stencil_store_op: VK_ATTACHMENT_STORE_OP_DONT_CARE,
        initial_layout: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        final_layout: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    let color_ref = VkAttachmentReference {
        attachment: 0,
        layout: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    let subpass = VkSubpassDescription {
        flags: 0,
        pipeline_bind_point: VK_PIPELINE_BIND_POINT_GRAPHICS,
        input_attachment_count: 0,
        p_input_attachments: core::ptr::null(),
        color_attachment_count: 1,
        p_color_attachments: &color_ref,
        p_resolve_attachments: core::ptr::null(),
        p_depth_stencil_attachment: core::ptr::null(),
        preserve_attachment_count: 0,
        p_preserve_attachments: core::ptr::null(),
    };
    let pass_info = VkRenderPassCreateInfo {
        s_type: VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        attachment_count: 1,
        p_attachments: &attachment,
        subpass_count: 1,
        p_subpasses: &subpass,
        dependency_count: 0,
        p_dependencies: core::ptr::null(),
    };
    // SAFETY: pass_info and its pointees outlive the call.
    check(
        unsafe {
            (fns.create_render_pass)(device, &pass_info, core::ptr::null(), &mut r.render_pass)
        },
        "vkCreateRenderPass",
    )?;

    // Descriptor layouts: set 0 the two atlases, set 1 one image.
    let atlas_bindings = [
        VkDescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: VK_SHADER_STAGE_FRAGMENT_BIT,
            p_immutable_samplers: core::ptr::null(),
        },
        VkDescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: VK_SHADER_STAGE_FRAGMENT_BIT,
            p_immutable_samplers: core::ptr::null(),
        },
    ];
    let atlas_info = VkDescriptorSetLayoutCreateInfo {
        s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        binding_count: atlas_bindings.len() as u32,
        p_bindings: atlas_bindings.as_ptr(),
    };
    // SAFETY: atlas_info and bindings outlive the call.
    check(
        unsafe {
            (fns.create_descriptor_set_layout)(
                device,
                &atlas_info,
                core::ptr::null(),
                &mut r.set_layout_atlas,
            )
        },
        "vkCreateDescriptorSetLayout (atlas)",
    )?;
    let image_binding = VkDescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
        stage_flags: VK_SHADER_STAGE_FRAGMENT_BIT,
        p_immutable_samplers: core::ptr::null(),
    };
    let image_info = VkDescriptorSetLayoutCreateInfo {
        s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        binding_count: 1,
        p_bindings: &image_binding,
    };
    // SAFETY: image_info and its binding outlive the call.
    check(
        unsafe {
            (fns.create_descriptor_set_layout)(
                device,
                &image_info,
                core::ptr::null(),
                &mut r.set_layout_image,
            )
        },
        "vkCreateDescriptorSetLayout (image)",
    )?;

    // Pipeline layout: both sets plus the viewport push constant.
    let set_layouts = [r.set_layout_atlas, r.set_layout_image];
    let push_range = VkPushConstantRange {
        stage_flags: VK_SHADER_STAGE_VERTEX_BIT,
        offset: 0,
        size: 8,
    };
    let layout_info = VkPipelineLayoutCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        set_layout_count: set_layouts.len() as u32,
        p_set_layouts: set_layouts.as_ptr(),
        push_constant_range_count: 1,
        p_push_constant_ranges: &push_range,
    };
    // SAFETY: layout_info and its pointees outlive the call.
    check(
        unsafe {
            (fns.create_pipeline_layout)(
                device,
                &layout_info,
                core::ptr::null(),
                &mut r.pipeline_layout,
            )
        },
        "vkCreatePipelineLayout",
    )?;

    // The pipeline, from the committed SPIR-V blobs.
    let vert_words = shader_words(include_bytes!("../shaders/quad.vert.spv"))?;
    let frag_words = shader_words(include_bytes!("../shaders/quad.frag.spv"))?;
    let mut vert_module: VkShaderModule = 0;
    let mut frag_module: VkShaderModule = 0;
    let vert_info = VkShaderModuleCreateInfo {
        s_type: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        code_size: vert_words.len() * 4,
        p_code: vert_words.as_ptr(),
    };
    // SAFETY: the words outlive the call.
    check(
        unsafe {
            (fns.create_shader_module)(device, &vert_info, core::ptr::null(), &mut vert_module)
        },
        "vkCreateShaderModule (vert)",
    )?;
    let frag_info = VkShaderModuleCreateInfo {
        s_type: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        code_size: frag_words.len() * 4,
        p_code: frag_words.as_ptr(),
    };
    // SAFETY: the words outlive the call.
    let fr = unsafe {
        (fns.create_shader_module)(device, &frag_info, core::ptr::null(), &mut frag_module)
    };
    if fr != VK_SUCCESS {
        // SAFETY: vert_module is live and unused.
        unsafe { (fns.destroy_shader_module)(device, vert_module, core::ptr::null()) };
        return Err(Error::msg(format!(
            "vkCreateShaderModule (frag) failed: VkResult {fr}"
        )));
    }

    let pipeline_result = create_pipeline(fns, device, r, vert_module, frag_module);
    // The modules are compiled into the pipeline; they are not needed after.
    // SAFETY: both modules are live; the pipeline (if any) keeps its own copy.
    unsafe {
        (fns.destroy_shader_module)(device, vert_module, core::ptr::null());
        (fns.destroy_shader_module)(device, frag_module, core::ptr::null());
    }
    pipeline_result?;

    // Sampler (texelFetch ignores filtering, but the binding needs one).
    let sampler_info = VkSamplerCreateInfo {
        s_type: VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        mag_filter: VK_FILTER_NEAREST,
        min_filter: VK_FILTER_NEAREST,
        mipmap_mode: 0,
        address_mode_u: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        address_mode_v: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        address_mode_w: VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        mip_lod_bias: 0.0,
        anisotropy_enable: 0,
        max_anisotropy: 1.0,
        compare_enable: 0,
        compare_op: 0,
        min_lod: 0.0,
        max_lod: 0.0,
        border_color: 0,
        unnormalized_coordinates: 0,
    };
    // SAFETY: sampler_info outlives the call.
    check(
        unsafe { (fns.create_sampler)(device, &sampler_info, core::ptr::null(), &mut r.sampler) },
        "vkCreateSampler",
    )?;

    // The descriptor pool and the two standing sets.
    let pool_size = VkDescriptorPoolSize {
        descriptor_type: VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        descriptor_count: DESCRIPTOR_MAX_SETS * 2,
    };
    let pool_info = VkDescriptorPoolCreateInfo {
        s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        max_sets: DESCRIPTOR_MAX_SETS,
        pool_size_count: 1,
        p_pool_sizes: &pool_size,
    };
    // SAFETY: pool_info outlives the call.
    check(
        unsafe {
            (fns.create_descriptor_pool)(
                device,
                &pool_info,
                core::ptr::null(),
                &mut r.descriptor_pool,
            )
        },
        "vkCreateDescriptorPool",
    )?;
    let alloc = VkDescriptorSetAllocateInfo {
        s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        p_next: core::ptr::null(),
        descriptor_pool: r.descriptor_pool,
        descriptor_set_count: 1,
        p_set_layouts: &r.set_layout_atlas,
    };
    // SAFETY: the pool and layout are live.
    check(
        unsafe { (fns.allocate_descriptor_sets)(device, &alloc, &mut r.set_atlas) },
        "vkAllocateDescriptorSets (atlas)",
    )?;
    let alloc_dummy = VkDescriptorSetAllocateInfo {
        s_type: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        p_next: core::ptr::null(),
        descriptor_pool: r.descriptor_pool,
        descriptor_set_count: 1,
        p_set_layouts: &r.set_layout_image,
    };
    // SAFETY: as above.
    check(
        unsafe { (fns.allocate_descriptor_sets)(device, &alloc_dummy, &mut r.set_dummy) },
        "vkAllocateDescriptorSets (dummy)",
    )?;
    Ok(())
}

/// The one graphics pipeline: the quad vertex layout, alpha blending matching
/// the software blitter, dynamic viewport/scissor.
fn create_pipeline(
    fns: &DeviceFns,
    device: VkDevice,
    r: &mut Renderer,
    vert: VkShaderModule,
    frag: VkShaderModule,
) -> Result<()> {
    let stages = [
        VkPipelineShaderStageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            stage: VK_SHADER_STAGE_VERTEX_BIT,
            module: vert,
            p_name: c"main".as_ptr(),
            p_specialization_info: core::ptr::null(),
        },
        VkPipelineShaderStageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            stage: VK_SHADER_STAGE_FRAGMENT_BIT,
            module: frag,
            p_name: c"main".as_ptr(),
            p_specialization_info: core::ptr::null(),
        },
    ];
    let binding = VkVertexInputBindingDescription {
        binding: 0,
        stride: size_of::<Vertex>() as u32,
        input_rate: VK_VERTEX_INPUT_RATE_VERTEX,
    };
    // Locations mirror shaders/quad.vert; offsets mirror gpu::Vertex.
    let attrs = [
        VkVertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: VK_FORMAT_R32G32_SFLOAT,
            offset: 0,
        },
        VkVertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: VK_FORMAT_R32G32_SFLOAT,
            offset: 8,
        },
        VkVertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: VK_FORMAT_R32G32B32A32_SFLOAT,
            offset: 16,
        },
        VkVertexInputAttributeDescription {
            location: 3,
            binding: 0,
            format: VK_FORMAT_R32G32B32A32_SFLOAT,
            offset: 32,
        },
        VkVertexInputAttributeDescription {
            location: 4,
            binding: 0,
            format: VK_FORMAT_R32_UINT,
            offset: 48,
        },
    ];
    let vertex_input = VkPipelineVertexInputStateCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        vertex_binding_description_count: 1,
        p_vertex_binding_descriptions: &binding,
        vertex_attribute_description_count: attrs.len() as u32,
        p_vertex_attribute_descriptions: attrs.as_ptr(),
    };
    let input_assembly = VkPipelineInputAssemblyStateCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        topology: VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        primitive_restart_enable: 0,
    };
    // Viewport and scissor are dynamic (set per frame for the window size).
    let viewport_state = VkPipelineViewportStateCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        viewport_count: 1,
        p_viewports: core::ptr::null(),
        scissor_count: 1,
        p_scissors: core::ptr::null(),
    };
    let raster = VkPipelineRasterizationStateCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        depth_clamp_enable: 0,
        rasterizer_discard_enable: 0,
        polygon_mode: VK_POLYGON_MODE_FILL,
        cull_mode: 0,
        front_face: 0,
        depth_bias_enable: 0,
        depth_bias_constant_factor: 0.0,
        depth_bias_clamp: 0.0,
        depth_bias_slope_factor: 0.0,
        line_width: 1.0,
    };
    let multisample = VkPipelineMultisampleStateCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        rasterization_samples: VK_SAMPLE_COUNT_1_BIT,
        sample_shading_enable: 0,
        min_sample_shading: 0.0,
        p_sample_mask: core::ptr::null(),
        alpha_to_coverage_enable: 0,
        alpha_to_one_enable: 0,
    };
    // Straight-alpha blending, the GPU twin of render.rs blend_over. Solid
    // quads carry alpha 1 and come out exact.
    let blend_attachment = VkPipelineColorBlendAttachmentState {
        blend_enable: 1,
        src_color_blend_factor: VK_BLEND_FACTOR_SRC_ALPHA,
        dst_color_blend_factor: VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        color_blend_op: VK_BLEND_OP_ADD,
        src_alpha_blend_factor: VK_BLEND_FACTOR_ONE,
        dst_alpha_blend_factor: VK_BLEND_FACTOR_ZERO,
        alpha_blend_op: VK_BLEND_OP_ADD,
        color_write_mask: VK_COLOR_COMPONENT_RGBA,
    };
    let blend = VkPipelineColorBlendStateCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        logic_op_enable: 0,
        logic_op: 0,
        attachment_count: 1,
        p_attachments: &blend_attachment,
        blend_constants: [0.0; 4],
    };
    let dynamic_states = [VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR];
    let dynamic = VkPipelineDynamicStateCreateInfo {
        s_type: VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        dynamic_state_count: dynamic_states.len() as u32,
        p_dynamic_states: dynamic_states.as_ptr(),
    };
    let info = VkGraphicsPipelineCreateInfo {
        s_type: VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        stage_count: stages.len() as u32,
        p_stages: stages.as_ptr(),
        p_vertex_input_state: &vertex_input,
        p_input_assembly_state: &input_assembly,
        p_tessellation_state: core::ptr::null(),
        p_viewport_state: &viewport_state,
        p_rasterization_state: &raster,
        p_multisample_state: &multisample,
        p_depth_stencil_state: core::ptr::null(),
        p_color_blend_state: &blend,
        p_dynamic_state: &dynamic,
        layout: r.pipeline_layout,
        render_pass: r.render_pass,
        subpass: 0,
        base_pipeline_handle: 0,
        base_pipeline_index: -1,
    };
    // SAFETY: info and every state struct it points at outlive the call.
    check(
        unsafe {
            (fns.create_graphics_pipelines)(device, 0, 1, &info, core::ptr::null(), &mut r.pipeline)
        },
        "vkCreateGraphicsPipelines",
    )
}

/// Fetch a device-level command by name (see [`load`]).
///
/// # Safety
/// `T` must be the exact PFN type of `name` per the Vulkan spec.
unsafe fn load_device<T>(gdpa: PfnGetDeviceProcAddr, device: VkDevice, name: &CStr) -> Result<T> {
    // SAFETY: gdpa is vkGetDeviceProcAddr for this device.
    let p = unsafe { gdpa(device, name.as_ptr()) }
        .ok_or_else(|| Error::msg(format!("vulkan device lacks {}", name.to_string_lossy())))?;
    debug_assert_eq!(size_of::<T>(), size_of::<PfnVoid>());
    // SAFETY: caller guarantees T is the command's true signature.
    Ok(unsafe { core::mem::transmute_copy::<PfnVoid, T>(&p) })
}

fn create_instance(gipa: PfnGetInstanceProcAddr) -> Result<Instance> {
    // SAFETY: vkCreateInstance is fetchable with a null instance.
    let create: PfnCreateInstance =
        unsafe { load(gipa, core::ptr::null_mut(), c"vkCreateInstance") }?;
    let app = VkApplicationInfo {
        s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
        p_next: core::ptr::null(),
        p_application_name: c"bnknotes".as_ptr(),
        application_version: 0,
        p_engine_name: core::ptr::null(),
        engine_version: 0,
        api_version: API_VERSION_1_2,
    };
    let info = VkInstanceCreateInfo {
        s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        p_application_info: &app,
        enabled_layer_count: 0,
        pp_enabled_layer_names: core::ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: core::ptr::null(),
    };
    let mut raw: VkInstance = core::ptr::null_mut();
    // SAFETY: info and app are fully initialized and outlive the call.
    check(
        unsafe { create(&info, core::ptr::null(), &mut raw) },
        "vkCreateInstance",
    )?;
    // Fetch the destructor immediately so every later error path tears the
    // instance down through Instance's Drop.
    let destroy: PfnDestroyInstance = unsafe { load(gipa, raw, c"vkDestroyInstance") }?;
    Ok(Instance { raw, destroy })
}

fn enumerate_physicals(
    enumerate: PfnEnumeratePhysicalDevices,
    instance: VkInstance,
) -> Result<Vec<VkPhysicalDevice>> {
    // The standard two-call pattern; loop in case a device appears between the
    // count and the fill (VK_INCOMPLETE).
    loop {
        let mut count = 0u32;
        // SAFETY: instance is live; null array means "count only".
        check(
            unsafe { enumerate(instance, &mut count, core::ptr::null_mut()) },
            "vkEnumeratePhysicalDevices (count)",
        )?;
        if count == 0 {
            return Err(Error::msg("no vulkan physical devices"));
        }
        let mut devices = vec![core::ptr::null_mut(); count as usize];
        // SAFETY: devices has room for count entries.
        let r = unsafe { enumerate(instance, &mut count, devices.as_mut_ptr()) };
        if r == VK_INCOMPLETE {
            continue;
        }
        check(r, "vkEnumeratePhysicalDevices")?;
        devices.truncate(count as usize);
        return Ok(devices);
    }
}

/// Read a NUL-terminated device or extension name out of a fixed C array.
fn c_name(bytes: &[c_char]) -> String {
    let unsigned: Vec<u8> = bytes.iter().map(|&c| c as u8).collect();
    let end = unsigned
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(unsigned.len());
    String::from_utf8_lossy(&unsigned[..end]).into_owned()
}

/// Whether `wanted` appears in the device's supported extension list.
fn has_extension(exts: &[VkExtensionProperties], wanted: &CStr) -> bool {
    exts.iter()
        .any(|e| c_name(&e.extension_name).as_bytes() == wanted.to_bytes())
}

/// Pick the physical device whose DRM identity matches `main_device` (or the
/// first capable device when `None`), and which offers a graphics queue and
/// every required extension. Returns the device, its name, and the graphics
/// queue family index.
fn pick_device(
    get_props2: PfnGetPhysicalDeviceProperties2,
    enumerate_exts: PfnEnumerateDeviceExtensionProperties,
    get_queue_families: PfnGetPhysicalDeviceQueueFamilyProperties,
    physicals: &[VkPhysicalDevice],
    main_device: Option<u64>,
) -> Result<(VkPhysicalDevice, String, u32, Option<u32>)> {
    let want = main_device.map(|dev| {
        (
            i64::from(crate::dmabuf::dev_major(dev)),
            i64::from(crate::dmabuf::dev_minor(dev)),
        )
    });
    let mut seen = Vec::new();

    for &physical in physicals {
        let mut drm = VkPhysicalDeviceDrmPropertiesEXT {
            s_type: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT,
            p_next: core::ptr::null_mut(),
            has_primary: 0,
            has_render: 0,
            primary_major: 0,
            primary_minor: 0,
            render_major: 0,
            render_minor: 0,
        };
        // SAFETY: zeroed VkPhysicalDeviceProperties is a valid out-parameter;
        // the struct layouts are guarded by the size tests below.
        let mut props = unsafe { core::mem::zeroed::<VkPhysicalDeviceProperties2>() };
        props.s_type = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

        let exts = enumerate_extensions(enumerate_exts, physical)?;
        let name;
        if has_extension(&exts, EXT_PHYSICAL_DEVICE_DRM) {
            props.p_next = (&mut drm as *mut VkPhysicalDeviceDrmPropertiesEXT).cast();
            // SAFETY: physical is live; props chains one correctly-typed struct.
            unsafe { get_props2(physical, &mut props) };
            name = c_name(&props.properties.device_name);
        } else if want.is_none() {
            // No compositor to match: the DRM identity is not needed.
            // SAFETY: physical is live; props chains nothing.
            unsafe { get_props2(physical, &mut props) };
            name = c_name(&props.properties.device_name);
        } else {
            // Without the DRM identity the device cannot be matched to the
            // compositor's; note it for the error message and move on.
            // SAFETY: physical is live; props chains nothing.
            unsafe { get_props2(physical, &mut props) };
            seen.push(format!(
                "{} (no VK_EXT_physical_device_drm)",
                c_name(&props.properties.device_name)
            ));
            continue;
        }

        if let Some((want_major, want_minor)) = want {
            let primary_matches = drm.has_primary != 0
                && (drm.primary_major, drm.primary_minor) == (want_major, want_minor);
            let render_matches = drm.has_render != 0
                && (drm.render_major, drm.render_minor) == (want_major, want_minor);
            if !primary_matches && !render_matches {
                seen.push(format!(
                    "{name} (drm {}:{} / {}:{})",
                    drm.primary_major, drm.primary_minor, drm.render_major, drm.render_minor
                ));
                continue;
            }
        }

        let missing: Vec<&str> = REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .filter(|e| !has_extension(&exts, e))
            .filter_map(|e| e.to_str().ok())
            .collect();
        if !missing.is_empty() {
            // A device the compositor named that cannot serve is a hard,
            // explainable failure; an unnamed candidate is just skipped.
            if want.is_some() {
                return Err(Error::msg(format!(
                    "device {name} matches the compositor's DRM device but lacks {}",
                    missing.join(", ")
                )));
            }
            seen.push(format!("{name} (lacks {})", missing.join(", ")));
            continue;
        }

        let Some(queue_family) = graphics_family(get_queue_families, physical) else {
            if want.is_some() {
                return Err(Error::msg(format!("device {name} has no graphics queue")));
            }
            seen.push(format!("{name} (no graphics queue)"));
            continue;
        };
        // The render node (no DRM master needed) is where explicit sync creates
        // its syncobjs; `None` when the device exposes no render node.
        let render_minor = (drm.has_render != 0).then_some(drm.render_minor as u32);
        return Ok((physical, name, queue_family, render_minor));
    }

    let target = match want {
        Some((major, minor)) => format!("matches DRM device {major}:{minor}"),
        None => "is capable".to_string(),
    };
    Err(Error::msg(format!(
        "no vulkan device {target} (saw: {})",
        seen.join("; ")
    )))
}

fn enumerate_extensions(
    enumerate_exts: PfnEnumerateDeviceExtensionProperties,
    physical: VkPhysicalDevice,
) -> Result<Vec<VkExtensionProperties>> {
    loop {
        let mut count = 0u32;
        // SAFETY: physical is live; null array means "count only".
        check(
            unsafe {
                enumerate_exts(
                    physical,
                    core::ptr::null(),
                    &mut count,
                    core::ptr::null_mut(),
                )
            },
            "vkEnumerateDeviceExtensionProperties (count)",
        )?;
        // SAFETY: zeroed VkExtensionProperties (a char array and a u32) is valid.
        let mut exts =
            vec![unsafe { core::mem::zeroed::<VkExtensionProperties>() }; count as usize];
        // SAFETY: exts has room for count entries.
        let r =
            unsafe { enumerate_exts(physical, core::ptr::null(), &mut count, exts.as_mut_ptr()) };
        if r == VK_INCOMPLETE {
            continue;
        }
        check(r, "vkEnumerateDeviceExtensionProperties")?;
        exts.truncate(count as usize);
        return Ok(exts);
    }
}

/// The first queue family with graphics support, if any.
fn graphics_family(
    get_queue_families: PfnGetPhysicalDeviceQueueFamilyProperties,
    physical: VkPhysicalDevice,
) -> Option<u32> {
    let mut count = 0u32;
    // SAFETY: physical is live; null array means "count only".
    unsafe { get_queue_families(physical, &mut count, core::ptr::null_mut()) };
    // SAFETY: zeroed VkQueueFamilyProperties (plain integers) is valid.
    let mut families =
        vec![unsafe { core::mem::zeroed::<VkQueueFamilyProperties>() }; count as usize];
    // SAFETY: families has room for count entries.
    unsafe { get_queue_families(physical, &mut count, families.as_mut_ptr()) };
    families
        .iter()
        .take(count as usize)
        .position(|f| f.queue_flags & VK_QUEUE_GRAPHICS_BIT != 0 && f.queue_count > 0)
        .map(|i| i as u32)
}

fn create_logical_device(
    create_device: PfnCreateDevice,
    gdpa: PfnGetDeviceProcAddr,
    physical: VkPhysicalDevice,
    queue_family: u32,
) -> Result<Device> {
    let priority = 1.0f32;
    let queue_info = VkDeviceQueueCreateInfo {
        s_type: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        queue_family_index: queue_family,
        queue_count: 1,
        p_queue_priorities: &priority,
    };
    let ext_ptrs: Vec<*const c_char> = REQUIRED_DEVICE_EXTENSIONS
        .iter()
        .map(|e| e.as_ptr())
        .collect();
    let info = VkDeviceCreateInfo {
        s_type: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        queue_create_info_count: 1,
        p_queue_create_infos: &queue_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: core::ptr::null(),
        enabled_extension_count: ext_ptrs.len() as u32,
        pp_enabled_extension_names: ext_ptrs.as_ptr(),
        p_enabled_features: core::ptr::null(),
    };
    let mut raw: VkDevice = core::ptr::null_mut();
    // SAFETY: physical is live; info and everything it points at outlive the call.
    check(
        unsafe { create_device(physical, &info, core::ptr::null(), &mut raw) },
        "vkCreateDevice",
    )?;
    // As with the instance: fetch the destructor first, so later failures
    // tear the device down through Drop.
    let destroy: PfnDestroyDevice = unsafe { load_device(gdpa, raw, c"vkDestroyDevice") }?;
    Ok(Device { raw, destroy })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transcription's one real hazard is a struct size drifting from the
    /// C ABI: the driver writes past what we allocated. These sizes are from
    /// vulkan_core.h on x86_64/arm64 (LP64).
    #[test]
    fn struct_sizes_match_the_c_abi() {
        assert_eq!(size_of::<VkPhysicalDeviceLimits>(), 504);
        assert_eq!(size_of::<VkPhysicalDeviceSparseProperties>(), 20);
        assert_eq!(size_of::<VkPhysicalDeviceProperties>(), 824);
        assert_eq!(size_of::<VkPhysicalDeviceProperties2>(), 840);
        assert_eq!(size_of::<VkPhysicalDeviceDrmPropertiesEXT>(), 56);
        assert_eq!(size_of::<VkQueueFamilyProperties>(), 24);
        assert_eq!(size_of::<VkExtensionProperties>(), 260);
        assert_eq!(size_of::<VkApplicationInfo>(), 48);
        assert_eq!(size_of::<VkInstanceCreateInfo>(), 64);
        assert_eq!(size_of::<VkDeviceQueueCreateInfo>(), 40);
        assert_eq!(size_of::<VkDeviceCreateInfo>(), 72);
        assert_eq!(size_of::<VkPhysicalDeviceMemoryProperties>(), 520);
        assert_eq!(size_of::<VkFormatProperties2>(), 32);
        assert_eq!(size_of::<VkDrmFormatModifierPropertiesEXT>(), 16);
        assert_eq!(size_of::<VkDrmFormatModifierPropertiesListEXT>(), 32);
        assert_eq!(size_of::<VkImageCreateInfo>(), 88);
        assert_eq!(size_of::<VkExternalMemoryImageCreateInfo>(), 24);
        assert_eq!(size_of::<VkImageDrmFormatModifierListCreateInfoEXT>(), 32);
        assert_eq!(size_of::<VkImageDrmFormatModifierPropertiesEXT>(), 24);
        assert_eq!(size_of::<VkMemoryRequirements>(), 24);
        assert_eq!(size_of::<VkMemoryAllocateInfo>(), 32);
        assert_eq!(size_of::<VkMemoryDedicatedAllocateInfo>(), 32);
        assert_eq!(size_of::<VkMemoryGetFdInfoKHR>(), 32);
        assert_eq!(size_of::<VkSubresourceLayout>(), 40);
        assert_eq!(size_of::<VkCommandBufferAllocateInfo>(), 32);
        assert_eq!(size_of::<VkImageMemoryBarrier>(), 72);
        assert_eq!(size_of::<VkSubmitInfo>(), 72);
        assert_eq!(size_of::<VkSemaphoreGetFdInfoKHR>(), 32);
        assert_eq!(size_of::<VkImportSemaphoreFdInfoKHR>(), 40);
    }

    #[test]
    fn memory_type_picking_prefers_device_local() {
        let gpu_types = [0u32, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0x6];
        // A fake Gpu is not constructible here (it needs a live device), so
        // test the picking logic through a copy of its rule.
        let pick = |allowed_bits: u32| -> Option<u32> {
            let allowed = |i: usize| allowed_bits & (1u32 << i) != 0;
            gpu_types
                .iter()
                .enumerate()
                .position(|(i, &flags)| {
                    allowed(i) && flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0
                })
                .or_else(|| (0..gpu_types.len()).find(|&i| allowed(i)))
                .map(|i| i as u32)
        };
        assert_eq!(pick(0b111), Some(1), "device-local wins");
        assert_eq!(pick(0b101), Some(0), "fallback to first allowed");
        assert_eq!(pick(0), None);
    }

    #[test]
    fn c_name_stops_at_the_nul() {
        let mut bytes = [0 as c_char; 8];
        for (i, b) in b"abc\0zzz".iter().enumerate() {
            bytes[i] = *b as c_char;
        }
        assert_eq!(c_name(&bytes), "abc");
        // No NUL at all: the whole array is the name.
        let full = [b'x' as c_char; 4];
        assert_eq!(c_name(&full), "xxxx");
    }

    #[test]
    fn api_version_is_1_2() {
        assert_eq!(API_VERSION_1_2 >> 22, 1);
        assert_eq!((API_VERSION_1_2 >> 12) & 0x3ff, 2);
    }
}
