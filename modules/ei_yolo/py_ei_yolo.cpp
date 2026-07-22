/*
 * Native OpenMV <-> Edge Impulse YOLO-Pro EON bridge for Arduino Nicla Vision.
 *
 * This module intentionally keeps the user's original EON-compiled model and
 * Edge Impulse post-processing. Python only passes an OpenMV image to native C++.
 *
 * Python API:
 *   import ei_yolo
 *   detections = ei_yolo.detect(img, min_confidence=0.50, debug=False)
 *
 * Each detection is returned as:
 *   (x, y, width, height, confidence, label)
 *
 * Coordinates are mapped back to the ORIGINAL OpenMV image after applying the
 * same square FIT_SHORTEST geometry used by the 96x96 model.
 */

extern "C" {
#include "py/runtime.h"
#include "py/obj.h"
#include "py/binary.h"
#include "py_image.h"
}

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <new>
#include <string.h>

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "model-parameters/model_metadata.h"


// OpenMV links native C++ modules without libstdc++. These are the
// small runtime functions required by the Edge Impulse classifier.
void *operator new[](size_t size) {
    return ei_malloc(size);
}

void operator delete[](void *ptr) noexcept {
    ei_free(ptr);
}

namespace std {
__attribute__((weak, noreturn)) void __throw_bad_function_call() {
    while (true) {
    }
}
} // namespace std

namespace {

static image_t *g_image = nullptr;
static int32_t g_crop_x = 0;
static int32_t g_crop_y = 0;
static int32_t g_crop_w = 0;
static int32_t g_crop_h = 0;


// OpenMV does not run C++ global constructors. Construct Edge Impulse's
// model handle lazily, after the firmware allocator is ready.
alignas(ei_impulse_handle_t)
static uint8_t g_impulse_handle_storage[sizeof(ei_impulse_handle_t)];
static ei_impulse_handle_t *g_impulse_handle = nullptr;

static ei_impulse_handle_t *get_impulse_handle(void) {
    if (g_impulse_handle == nullptr) {
        g_impulse_handle = new (g_impulse_handle_storage)
            ei_impulse_handle_t(&impulse_1042434_1);
    }
    return g_impulse_handle;
}

static inline uint8_t clamp_u8(int32_t value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return (uint8_t)value;
}

static inline uint16_t read_rgb565(const image_t *img, int32_t x, int32_t y) {
    return IMAGE_GET_RGB565_PIXEL(img, x, y);
}

static inline void rgb565_to_rgb888(uint16_t pixel, int32_t &r, int32_t &g, int32_t &b) {
    r = COLOR_RGB565_TO_R8(pixel);
    g = COLOR_RGB565_TO_G8(pixel);
    b = COLOR_RGB565_TO_B8(pixel);
}

/*
 * Edge Impulse image signals use one float per pixel. For RGB models the float
 * contains a packed 0xRRGGBB value. This callback creates the model's 96x96
 * square input directly from the OpenMV RGB565 frame without a second image buffer.
 *
 * FIT_SHORTEST for a square model is equivalent to:
 *   center-crop the source to a square -> resize square to 96x96.
 *
 * Bilinear interpolation is done here to stay close to EI's image preprocessing.
 */
static int ei_yolo_get_data(size_t offset, size_t length, float *out_ptr) {
    if (g_image == nullptr || !IM_IS_RGB565(g_image)) {
        return -1;
    }

    const int32_t dst_w = EI_CLASSIFIER_INPUT_WIDTH;
    const int32_t dst_h = EI_CLASSIFIER_INPUT_HEIGHT;

    for (size_t i = 0; i < length; ++i) {
        const size_t index = offset + i;
        const int32_t dx = (int32_t)(index % (size_t)dst_w);
        const int32_t dy = (int32_t)(index / (size_t)dst_w);

        // Map destination pixel centers to the cropped source image.
        const float sx_f = g_crop_x + (((float)dx + 0.5f) * (float)g_crop_w / (float)dst_w) - 0.5f;
        const float sy_f = g_crop_y + (((float)dy + 0.5f) * (float)g_crop_h / (float)dst_h) - 0.5f;

        int32_t x0 = (int32_t)floorf(sx_f);
        int32_t y0 = (int32_t)floorf(sy_f);
        float fx = sx_f - (float)x0;
        float fy = sy_f - (float)y0;

        if (x0 < g_crop_x) { x0 = g_crop_x; fx = 0.0f; }
        if (y0 < g_crop_y) { y0 = g_crop_y; fy = 0.0f; }

        int32_t x1 = x0 + 1;
        int32_t y1 = y0 + 1;
        const int32_t crop_x_max = g_crop_x + g_crop_w - 1;
        const int32_t crop_y_max = g_crop_y + g_crop_h - 1;
        if (x0 > crop_x_max) x0 = crop_x_max;
        if (y0 > crop_y_max) y0 = crop_y_max;
        if (x1 > crop_x_max) x1 = crop_x_max;
        if (y1 > crop_y_max) y1 = crop_y_max;

        int32_t r00, g00, b00;
        int32_t r10, g10, b10;
        int32_t r01, g01, b01;
        int32_t r11, g11, b11;
        rgb565_to_rgb888(read_rgb565(g_image, x0, y0), r00, g00, b00);
        rgb565_to_rgb888(read_rgb565(g_image, x1, y0), r10, g10, b10);
        rgb565_to_rgb888(read_rgb565(g_image, x0, y1), r01, g01, b01);
        rgb565_to_rgb888(read_rgb565(g_image, x1, y1), r11, g11, b11);

        const float inv_fx = 1.0f - fx;
        const float inv_fy = 1.0f - fy;

        const int32_t r = (int32_t)roundf(
            (r00 * inv_fx * inv_fy) + (r10 * fx * inv_fy) +
            (r01 * inv_fx * fy) + (r11 * fx * fy));
        const int32_t g = (int32_t)roundf(
            (g00 * inv_fx * inv_fy) + (g10 * fx * inv_fy) +
            (g01 * inv_fx * fy) + (g11 * fx * fy));
        const int32_t b = (int32_t)roundf(
            (b00 * inv_fx * inv_fy) + (b10 * fx * inv_fy) +
            (b01 * inv_fx * fy) + (b11 * fx * fy));

        const uint32_t packed = ((uint32_t)clamp_u8(r) << 16)
                              | ((uint32_t)clamp_u8(g) << 8)
                              | ((uint32_t)clamp_u8(b));
        out_ptr[i] = (float)packed;
    }

    return 0;
}

static void setup_square_fit_shortest(image_t *img) {
    g_image = img;

    if (img->w > img->h) {
        g_crop_w = img->h;
        g_crop_h = img->h;
        g_crop_x = (img->w - img->h) / 2;
        g_crop_y = 0;
    }
    else {
        g_crop_w = img->w;
        g_crop_h = img->w;
        g_crop_x = 0;
        g_crop_y = (img->h - img->w) / 2;
    }
}

static int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

} // namespace

static mp_obj_t py_ei_yolo_info(void) {
    mp_obj_t dict = mp_obj_new_dict(0);
    mp_obj_dict_store(dict, MP_OBJ_NEW_QSTR(MP_QSTR_input_width), mp_obj_new_int(EI_CLASSIFIER_INPUT_WIDTH));
    mp_obj_dict_store(dict, MP_OBJ_NEW_QSTR(MP_QSTR_input_height), mp_obj_new_int(EI_CLASSIFIER_INPUT_HEIGHT));
    mp_obj_dict_store(dict, MP_OBJ_NEW_QSTR(MP_QSTR_input_channels), mp_obj_new_int(3));
    mp_obj_dict_store(dict, MP_OBJ_NEW_QSTR(MP_QSTR_eon_compiled), mp_const_true);
    mp_obj_dict_store(dict, MP_OBJ_NEW_QSTR(MP_QSTR_default_threshold), mp_obj_new_float(EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD));
    mp_obj_dict_store(dict, MP_OBJ_NEW_QSTR(MP_QSTR_label), mp_obj_new_str("0", 1));
    return dict;
}
static MP_DEFINE_CONST_FUN_OBJ_0(py_ei_yolo_info_obj, py_ei_yolo_info);

enum {
    ARG_img,
    ARG_min_confidence,
    ARG_debug,
};

static mp_obj_t py_ei_yolo_detect(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_img,            MP_ARG_REQUIRED | MP_ARG_OBJ,  {.u_obj = MP_OBJ_NULL} },
        { MP_QSTR_min_confidence, MP_ARG_KW_ONLY  | MP_ARG_OBJ,  {.u_obj = MP_OBJ_NULL} },
        { MP_QSTR_debug,          MP_ARG_KW_ONLY  | MP_ARG_BOOL, {.u_bool = false} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    image_t *img = (image_t *)py_image_cobj(args[ARG_img].u_obj);
    if (img == nullptr) {
        mp_raise_ValueError(MP_ERROR_TEXT("invalid image"));
    }
    if (!IM_IS_RGB565(img)) {
        mp_raise_ValueError(MP_ERROR_TEXT("ei_yolo requires an RGB565 image"));
    }
    if (img->w <= 0 || img->h <= 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("image has invalid dimensions"));
    }

    float min_confidence = EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD;
    if (args[ARG_min_confidence].u_obj != MP_OBJ_NULL) {
        min_confidence = mp_obj_get_float(args[ARG_min_confidence].u_obj);
    }

    // The model's native postprocessor already removes boxes below 0.50.
    // A higher Python threshold is allowed; a lower one cannot restore removed boxes.
    if (min_confidence < EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD) {
        min_confidence = EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD;
    }
    if (min_confidence > 1.0f) {
        min_confidence = 1.0f;
    }

    setup_square_fit_shortest(img);

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &ei_yolo_get_data;

    ei_impulse_result_t result = { 0 };
    const EI_IMPULSE_ERROR error = process_impulse(
        get_impulse_handle(), &signal, &result, args[ARG_debug].u_bool);

    // Stop exposing the image through global callback state immediately after inference.
    g_image = nullptr;

    if (error != EI_IMPULSE_OK) {
        mp_raise_msg_varg(&mp_type_RuntimeError,
            MP_ERROR_TEXT("Edge Impulse inference failed (%d)"), (int)error);
    }

    mp_obj_t list = mp_obj_new_list(0, nullptr);

    const float scale_x = (float)g_crop_w / (float)EI_CLASSIFIER_INPUT_WIDTH;
    const float scale_y = (float)g_crop_h / (float)EI_CLASSIFIER_INPUT_HEIGHT;

    for (uint32_t i = 0; i < result.bounding_boxes_count; ++i) {
        const ei_impulse_result_bounding_box_t &bb = result.bounding_boxes[i];
        if (bb.value < min_confidence || bb.width == 0 || bb.height == 0) {
            continue;
        }

        int32_t x = g_crop_x + (int32_t)roundf((float)bb.x * scale_x);
        int32_t y = g_crop_y + (int32_t)roundf((float)bb.y * scale_y);
        int32_t w = (int32_t)roundf((float)bb.width * scale_x);
        int32_t h = (int32_t)roundf((float)bb.height * scale_y);

        x = clamp_i32(x, 0, img->w - 1);
        y = clamp_i32(y, 0, img->h - 1);
        w = clamp_i32(w, 1, img->w - x);
        h = clamp_i32(h, 1, img->h - y);

        mp_obj_t tuple_items[6] = {
            mp_obj_new_int(x),
            mp_obj_new_int(y),
            mp_obj_new_int(w),
            mp_obj_new_int(h),
            mp_obj_new_float(bb.value),
            mp_obj_new_str(bb.label ? bb.label : "0", bb.label ? strlen(bb.label) : 1),
        };
        mp_obj_list_append(list, mp_obj_new_tuple(6, tuple_items));
    }

    return list;
}
static MP_DEFINE_CONST_FUN_OBJ_KW(py_ei_yolo_detect_obj, 1, py_ei_yolo_detect);

static const mp_rom_map_elem_t ei_yolo_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_ei_yolo) },
    { MP_ROM_QSTR(MP_QSTR_detect), MP_ROM_PTR(&py_ei_yolo_detect_obj) },
    { MP_ROM_QSTR(MP_QSTR_info), MP_ROM_PTR(&py_ei_yolo_info_obj) },
};
static MP_DEFINE_CONST_DICT(ei_yolo_module_globals, ei_yolo_module_globals_table);


// TensorFlow Lite may call abort(), which Newlib connects to these process
// functions. Nicla Vision has no operating-system processes, so use safe stubs.
extern "C" __attribute__((weak)) int _kill(int pid, int signal) {
    (void)pid;
    (void)signal;
    return -1;
}

extern "C" __attribute__((weak)) int _getpid(void) {
    return 1;
}

extern "C" const mp_obj_module_t ei_yolo_user_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&ei_yolo_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_ei_yolo, ei_yolo_user_cmodule);
