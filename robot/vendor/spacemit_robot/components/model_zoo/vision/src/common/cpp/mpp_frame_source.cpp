/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mpp_frame_source.h"

#include <cstring>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#ifdef VISION_HAS_MPP
#include <unistd.h>
#include <dlfcn.h>
#ifdef MIN
#undef MIN
#endif
extern "C" {
#include "sys_api.h"
#include "vb_api.h"
#include "uvc_api.h"
#include "vi_api.h"
}
#include "vdec/vdec_api.h"
#include "v2d_api.h"
#endif  // VISION_HAS_MPP

namespace vision_mpp {

namespace {

std::string video_device_path(const std::string& v4l2_dev, int camera_id) {
    if (!v4l2_dev.empty()) return v4l2_dev;
    return "/dev/video" + std::to_string(camera_id);
}

#ifdef VISION_HAS_MPP

MppPixelFormat parse_pixel_format(const std::string& fmt) {
    if (fmt == "YUYV" || fmt == "yuyv" || fmt == "YUV2") return MPP_PIXEL_FORMAT_YUYV;
    if (fmt == "NV12" || fmt == "nv12") return MPP_PIXEL_FORMAT_NV12;
    return MPP_PIXEL_FORMAT_MJPEG;
}

bool trim_mjpeg_span(const uchar* data, size_t size, size_t* out_off, size_t* out_len) {
    if (data == nullptr || size < 4 || out_off == nullptr || out_len == nullptr) return false;
    size_t soi = size;
    for (size_t i = 0; i + 1 < size; ++i) {
        if (data[i] == 0xFF && data[i + 1] == 0xD8) { soi = i; break; }
    }
    if (soi >= size) return false;
    size_t eoi = size;
    for (size_t i = size; i > soi + 1; --i) {
        if (data[i - 2] == 0xFF && data[i - 1] == 0xD9) { eoi = i; break; }
    }
    if (eoi <= soi + 2) return false;
    *out_off = soi;
    *out_len = eoi - soi;
    return true;
}

bool nv12_frame_to_bgr_cpu(const VideoFrameInfo& frame, cv::Mat* bgr) {
    if (bgr == nullptr) return false;
    U32 w = frame.stVdecFrameInfo.stCommFrameInfo.u32Width;
    U32 h = frame.stVdecFrameInfo.stCommFrameInfo.u32Height;
    if (w == 0 || h == 0) {
        w = frame.stCommFrameInfo.u32Width;
        h = frame.stCommFrameInfo.u32Height;
    }
    if (w == 0 || h == 0 || frame.stVFrame.u32PlaneNum < 2) return false;
    auto* y_ptr = reinterpret_cast<uchar*>(frame.stVFrame.ulPlaneVirAddr[0]);
    auto* uv_ptr = reinterpret_cast<uchar*>(frame.stVFrame.ulPlaneVirAddr[1]);
    if (y_ptr == nullptr || uv_ptr == nullptr) return false;

    const int iw = static_cast<int>(w);
    const int ih = static_cast<int>(h);
    int y_stride = static_cast<int>(frame.stVFrame.u32PlaneStride[0]);
    int uv_stride = static_cast<int>(frame.stVFrame.u32PlaneStride[1]);
    if (y_stride <= 0) y_stride = iw;
    if (uv_stride <= 0) uv_stride = iw;

    MppPixelFormat pix = frame.stVdecFrameInfo.stCommFrameInfo.ePixelFormat;
    if (pix == MPP_PIXEL_FORMAT_UNKNOWN) pix = frame.stCommFrameInfo.ePixelFormat;

    cv::Mat y(ih, iw, CV_8UC1, y_ptr, y_stride);
    cv::Mat uv(ih / 2, iw / 2, CV_8UC2, uv_ptr, uv_stride);
    const int cvt_code = (pix == MPP_PIXEL_FORMAT_NV21) ?
        cv::COLOR_YUV2BGR_NV21 : cv::COLOR_YUV2BGR_NV12;
    cv::cvtColorTwoPlane(y, uv, *bgr, cvt_code);
    return !bgr->empty();
}

constexpr long kV2dPixelThreshold = 1280L * 720L;

void normalize_vdec_nv12_frame(VideoFrameInfo* frame) {
    if (frame == nullptr) return;
    if (frame->stCommFrameInfo.u32Width == 0 || frame->stCommFrameInfo.u32Height == 0) {
        frame->stCommFrameInfo = frame->stVdecFrameInfo.stCommFrameInfo;
    }
    if (frame->stCommFrameInfo.ePixelFormat == MPP_PIXEL_FORMAT_UNKNOWN) {
        frame->stCommFrameInfo.ePixelFormat = MPP_PIXEL_FORMAT_NV12;
    }
}

bool prepare_vb_bgr_frame(U32 width, U32 height, UL* pool_id, VideoFrameInfo* frame) {
    if (pool_id == nullptr || frame == nullptr) return false;
    const U32 stride = width * 3U;
    const U32 y_size = stride * height;

    VbPoolCfg cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.u32BufSize = y_size;
    cfg.u32BufCnt = 1;
    cfg.eModId = MPP_ID_V2D;
    cfg.eRemapMode = VBUF_REMAP_MODE_NOCACHE;

    *pool_id = VB_CreatePool(&cfg);
    if (*pool_id == 0) return false;

    const UL buffer = VB_GetBuffer(*pool_id, 0);
    if (buffer == 0) {
        VB_DestroyPool(*pool_id);
        *pool_id = 0;
        return false;
    }

    S32 dma_fd = -1;
    void* vir = nullptr;
    if (VB_GetDmaBufFd(buffer, &dma_fd) != 0 || dma_fd < 0 ||
        VB_GetVirAddr(buffer, &vir) != 0 || vir == nullptr) {
        VB_ReleaseBuffer(buffer);
        VB_DestroyPool(*pool_id);
        *pool_id = 0;
        return false;
    }

    std::memset(frame, 0, sizeof(*frame));
    frame->eFrameType = FRAME_TYPE_COMMON;
    frame->eModId = MPP_ID_V2D;
    frame->ulPoolId = *pool_id;
    frame->ulBufferId = buffer;
    frame->stCommFrameInfo.u32Width = width;
    frame->stCommFrameInfo.u32Height = height;
    frame->stCommFrameInfo.ePixelFormat = MPP_PIXEL_FORMAT_BGR_888;
    frame->stVFrame.u32PlaneNum = 1;
    frame->stVFrame.u32PlaneStride[0] = stride;
    frame->stVFrame.u32PlaneSize[0] = y_size;
    frame->stVFrame.u32PlaneSizeValid[0] = y_size;
    frame->stVFrame.u32TotalSize = y_size;
    frame->stVFrame.u32Fd[0] = static_cast<UL>(dma_fd);
    frame->stVFrame.ulPlaneVirAddr[0] = reinterpret_cast<UL>(vir);
    return true;
}

bool v2d_nv12_to_bgr(const VideoFrameInfo& src_nv12, VideoFrameInfo& dst_bgr, cv::Mat* bgr) {
    if (bgr == nullptr) return false;
    VideoFrameInfo src = src_nv12;
    normalize_vdec_nv12_frame(&src);

    V2DHandle handle = 0;
    if (V2D_BeginJob(&handle) != 0) return false;
    if (V2D_ConvertFrame(handle, &src, &dst_bgr) != 0) {
        (void)V2D_CancelJob(handle);
        return false;
    }
    if (V2D_EndJob(handle) != 0) return false;

    const U32 w = dst_bgr.stCommFrameInfo.u32Width;
    const U32 h = dst_bgr.stCommFrameInfo.u32Height;
    const U32 stride = dst_bgr.stVFrame.u32PlaneStride[0];
    auto* ptr = reinterpret_cast<uchar*>(dst_bgr.stVFrame.ulPlaneVirAddr[0]);
    if (ptr == nullptr || w == 0 || h == 0) return false;
    cv::Mat wrapped(static_cast<int>(h), static_cast<int>(w), CV_8UC3, ptr, stride);
    *bgr = wrapped.clone();
    return !bgr->empty();
}

void preload_vi_plugin() {
#ifdef VISION_MPP_STAGING_LIB
    const std::string dir = VISION_MPP_STAGING_LIB;
    if (!dir.empty() && access(dir.c_str(), F_OK) == 0) {
        const char* old = std::getenv("LD_LIBRARY_PATH");
        std::string merged = dir;
        if (old != nullptr && old[0] != '\0') { merged += ":"; merged += old; }
        (void)setenv("LD_LIBRARY_PATH", merged.c_str(), 1);
    }
    for (const char* name : {"/libvi_k3_cam_plugin.so", "/libvi_k1_cam_plugin.so"}) {
        const std::string so = dir + name;
        if (access(so.c_str(), F_OK) != 0) continue;
        if (dlopen(so.c_str(), RTLD_LAZY | RTLD_GLOBAL) != nullptr) {
            std::cout << "Preloaded MPP VI plugin: " << so << "\n";
            return;
        }
    }
#endif
}

#endif  // VISION_HAS_MPP

}  // namespace

struct MppFrameSource::Impl {
    explicit Impl(const MppFrameSourceConfig& c) : cfg(c) {}

    MppFrameSourceConfig cfg;

    cv::VideoCapture cap;
    bool cap_opened = false;

    bool open_opencv() {
        const std::string node = video_device_path(cfg.v4l2_dev, cfg.camera_id);
        if (!cfg.v4l2_dev.empty()) {
            cap.open(cfg.v4l2_dev);
        } else {
            cap.open(cfg.camera_id);
        }
        cap_opened = cap.isOpened();
        if (!cap_opened) {
            std::cerr << "MppFrameSource: cv::VideoCapture failed to open " << node << "\n";
        }
        return cap_opened;
    }

    bool read_opencv(cv::Mat* out_bgr) {
        return cap_opened && cap.read(*out_bgr) && !out_bgr->empty();
    }

#ifdef VISION_HAS_MPP
    bool mpp_active = false;

    bool sys_inited = false, vb_inited = false, uvc_inited = false;
    bool dev_created = false, dev_enabled = false, chn_enabled = false;
    bool vdec_inited = false, vdec_created = false, vdec_enabled = false;
    S32  vdec_chn = 0;
    MppPixelFormat pixel_format = MPP_PIXEL_FORMAT_MJPEG;

    bool vi_inited = false, vi_dev_enabled = false, vi_chn_enabled = false;

    bool v2d_ready = false;
    UL   bgr_pool = 0;
    VideoFrameInfo bgr_frame{};

    void maybe_init_v2d(int w, int h) {
        if (cfg.cpu_color) return;
        if (static_cast<long>(w) * h < kV2dPixelThreshold) return;
        if (prepare_vb_bgr_frame(static_cast<U32>(w), static_cast<U32>(h), &bgr_pool, &bgr_frame)) {
            v2d_ready = true;
            std::cout << "MPP NV12->BGR: V2D (>= " << kV2dPixelThreshold << " px)\n";
        } else {
            std::cerr << "MPP NV12->BGR: V2D pool init failed, using CPU cvtColor\n";
        }
    }

    bool nv12_to_bgr(const VideoFrameInfo& nv12, cv::Mat* out_bgr) {
        if (v2d_ready && v2d_nv12_to_bgr(nv12, bgr_frame, out_bgr)) return true;
        return nv12_frame_to_bgr_cpu(nv12, out_bgr);
    }

    void release_v2d() {
        if (bgr_frame.ulBufferId != 0) { (void)VB_ReleaseBuffer(bgr_frame.ulBufferId); bgr_frame.ulBufferId = 0; }
        if (bgr_pool != 0) { (void)VB_DestroyPool(bgr_pool); bgr_pool = 0; }
        v2d_ready = false;
    }

    bool open_uvc() {
        pixel_format = parse_pixel_format(cfg.format);
        if (SYS_Init() != 0) { std::cerr << "MPP UVC: SYS_Init failed\n"; return false; }
        sys_inited = true;
        if (VB_Init() != 0) { std::cerr << "MPP UVC: VB_Init failed\n"; return false; }
        vb_inited = true;
        if (UVC_Init() != 0) { std::cerr << "MPP UVC: UVC_Init failed\n"; return false; }
        uvc_inited = true;

        UvcDevAttr dev_attr;
        std::memset(&dev_attr, 0, sizeof(dev_attr));
        const std::string node = video_device_path(cfg.v4l2_dev, cfg.camera_id);
        std::strncpy(dev_attr.acDevNode, node.c_str(), sizeof(dev_attr.acDevNode) - 1);
        if (UVC_CreateDev(0, &dev_attr) != 0) { std::cerr << "MPP UVC: CreateDev failed\n"; return false; }
        dev_created = true;
        if (UVC_EnableDev(0) != 0) { std::cerr << "MPP UVC: EnableDev failed\n"; return false; }
        dev_enabled = true;

        UvcChnAttr chn_attr;
        std::memset(&chn_attr, 0, sizeof(chn_attr));
        chn_attr.u32Width = static_cast<U32>(cfg.width);
        chn_attr.u32Height = static_cast<U32>(cfg.height);
        chn_attr.ePixelFormat = pixel_format;
        chn_attr.u32Fps = static_cast<U32>(cfg.fps);
        chn_attr.u32Depth = 1;
        if (UVC_SetChnAttr(0, 0, &chn_attr) != 0) { std::cerr << "MPP UVC: SetChnAttr failed\n"; return false; }
        if (UVC_EnableChn(0, 0) != 0) { std::cerr << "MPP UVC: EnableChn failed\n"; return false; }
        chn_enabled = true;

        if (pixel_format == MPP_PIXEL_FORMAT_MJPEG) {
            if (VDEC_Init() != 0) { std::cerr << "MPP UVC: VDEC_Init failed\n"; return false; }
            vdec_inited = true;
            VdecChnAttr vdec_attr;
            std::memset(&vdec_attr, 0, sizeof(vdec_attr));
            vdec_attr.eCodecType = MPP_STREAM_CODEC_MJPEG;
            vdec_attr.eOutputPixelFormat = MPP_PIXEL_FORMAT_NV12;
            vdec_attr.u32Width = static_cast<U32>(cfg.width);
            vdec_attr.u32Height = static_cast<U32>(cfg.height);
            if (VDEC_CreateChn(vdec_chn, &vdec_attr) != 0) { std::cerr << "MPP UVC: VDEC_CreateChn failed\n"; return false; }
            vdec_created = true;
            if (VDEC_EnableChn(vdec_chn) != 0) { std::cerr << "MPP UVC: VDEC_EnableChn failed\n"; return false; }
            vdec_enabled = true;
            std::cout << "MPP pipeline: UVC(MJPEG) -> VDEC -> NV12 -> BGR\n";
        } else {
            std::cout << "MPP pipeline: UVC(" << cfg.format << ") -> NV12/YUYV -> BGR\n";
        }

        if (pixel_format != MPP_PIXEL_FORMAT_YUYV) {
            maybe_init_v2d(cfg.width, cfg.height);
        }

        for (int i = 0; i < 15; ++i) {
            cv::Mat discard;
            (void)read_uvc(&discard);
        }
        return true;
    }

    bool read_uvc(cv::Mat* out_bgr) {
        if (out_bgr == nullptr || !chn_enabled) return false;

        VideoFrameInfo uvc_frame;
        std::memset(&uvc_frame, 0, sizeof(uvc_frame));
        if (UVC_GetFrame(0, 0, &uvc_frame, static_cast<S32>(cfg.timeout_ms)) != 0) return false;

        const auto fmt = uvc_frame.stCommFrameInfo.ePixelFormat;
        const int w = static_cast<int>(uvc_frame.stCommFrameInfo.u32Width);
        const int h = static_cast<int>(uvc_frame.stCommFrameInfo.u32Height);
        bool ok = false;

        if (fmt == MPP_PIXEL_FORMAT_MJPEG && vdec_enabled &&
            uvc_frame.stVFrame.ulPlaneVirAddr[0] != 0) {
            const uchar* raw = reinterpret_cast<const uchar*>(uvc_frame.stVFrame.ulPlaneVirAddr[0]);
            const size_t raw_size = uvc_frame.stVFrame.u32PlaneSizeValid[0];
            size_t off = 0, len = raw_size;
            std::vector<uchar> jpeg_storage;
            const uchar* stream_ptr = raw;
            if (trim_mjpeg_span(raw, raw_size, &off, &len)) {
                if (off > 0 || len < raw_size) {
                    jpeg_storage.assign(raw + off, raw + off + len);
                    stream_ptr = jpeg_storage.data();
                }
            }

            StreamBufferInfo stream;
            std::memset(&stream, 0, sizeof(stream));
            stream.pu8Addr = stream_ptr;
            stream.u32Size = static_cast<U32>(len);
            stream.eCodecType = MPP_STREAM_CODEC_MJPEG;
            stream.bKeyFrame = MPP_TRUE;
            stream.bEndOfStream = MPP_FALSE;
            stream.u64PTS = uvc_frame.stVFrame.u64PTS;

            const S32 send_ret = VDEC_SendStream(vdec_chn, &stream, static_cast<U32>(cfg.timeout_ms));
            (void)UVC_ReleaseFrame(0, 0, &uvc_frame);
            if (send_ret != 0 && send_ret != ERR_VDEC_EOS) return false;

            VideoFrameInfo dec_frame;
            std::memset(&dec_frame, 0, sizeof(dec_frame));
            const S32 dec_ret = VDEC_GetFrame(vdec_chn, &dec_frame, static_cast<U32>(cfg.timeout_ms));
            if (dec_ret != ERR_VDEC_OK) return false;
            ok = nv12_to_bgr(dec_frame, out_bgr);
            (void)VDEC_ReleaseFrame(vdec_chn, dec_frame.ulBufferId);
            return ok;
        }

        if (fmt == MPP_PIXEL_FORMAT_MJPEG) {
            (void)UVC_ReleaseFrame(0, 0, &uvc_frame);
            std::cerr << "MPP UVC: MJPEG requires VDEC; channel not decoding\n";
            return false;
        }

        if (fmt == MPP_PIXEL_FORMAT_YUYV && uvc_frame.stVFrame.ulPlaneVirAddr[0] != 0) {
            // Honor the plane stride: YUYV rows may be padded, so wrapping with
            // the default width*2 step would mis-offset later rows.
            int yuyv_stride = static_cast<int>(uvc_frame.stVFrame.u32PlaneStride[0]);
            if (yuyv_stride <= 0) yuyv_stride = w * 2;
            void* yuyv_ptr = reinterpret_cast<void*>(uvc_frame.stVFrame.ulPlaneVirAddr[0]);
            cv::Mat yuyv(h, w, CV_8UC2, yuyv_ptr, static_cast<size_t>(yuyv_stride));
            cv::cvtColor(yuyv, *out_bgr, cv::COLOR_YUV2BGR_YUYV);
            ok = !out_bgr->empty();
        } else if (fmt == MPP_PIXEL_FORMAT_NV12 || fmt == MPP_PIXEL_FORMAT_NV21) {
            const bool has_planes = uvc_frame.stVFrame.u32PlaneNum >= 2 &&
                uvc_frame.stVFrame.ulPlaneVirAddr[0] != 0 &&
                uvc_frame.stVFrame.ulPlaneVirAddr[1] != 0;
            if (has_planes) {
                ok = nv12_to_bgr(uvc_frame, out_bgr);
            }
        }
        (void)UVC_ReleaseFrame(0, 0, &uvc_frame);
        return ok;
    }

    bool open_vi() {
        preload_vi_plugin();
        const std::string node = video_device_path(cfg.v4l2_dev, cfg.camera_id);
        (void)setenv("K3_V4L2_DEV", node.c_str(), 1);

        if (SYS_Init() != 0) { std::cerr << "MPP VI: SYS_Init failed\n"; return false; }
        sys_inited = true;
        if (VB_Init() != 0) { std::cerr << "MPP VI: VB_Init failed\n"; return false; }
        vb_inited = true;
        if (VI_Init() != 0) {
            std::cerr << "MPP VI: VI_Init failed (check libvi_k3_cam_plugin.so)\n";
            return false;
        }
        vi_inited = true;

        ViDevAttrS dev_attr;
        std::memset(&dev_attr, 0, sizeof(dev_attr));
        dev_attr.eWorkMode = VI_WORK_MODE_ONLINE;
        dev_attr.u32Width = static_cast<U32>(cfg.sensor_width);
        dev_attr.u32Height = static_cast<U32>(cfg.sensor_height);
        dev_attr.u32MipiLaneNum = static_cast<U32>(cfg.mipi_lanes);
        dev_attr.u32mbps = static_cast<U32>(cfg.mbps);
        dev_attr.bCapture2Preview = MPP_FALSE;
        if (VI_SetDevAttr(0, &dev_attr) != 0) { std::cerr << "MPP VI: SetDevAttr failed\n"; return false; }

        ViChnAttrS chn_attr;
        std::memset(&chn_attr, 0, sizeof(chn_attr));
        chn_attr.eChnType = VI_CHN_TYPE_PHYSICAL;
        chn_attr.ePixelFormat = MPP_PIXEL_FORMAT_NV12;
        chn_attr.u32Width = static_cast<U32>(cfg.width);
        chn_attr.u32Height = static_cast<U32>(cfg.height);
        chn_attr.bMirror = MPP_FALSE;
        chn_attr.bFlip = MPP_FALSE;
        chn_attr.eRotateMode = VI_ROT_0;
        chn_attr.bCropEnable = MPP_FALSE;
        chn_attr.eStrideAlign = VI_STRIDE_ALIGN_DEFAULT;
        if (VI_SetChnAttr(0, static_cast<VI_CHN>(cfg.vi_chn), &chn_attr) != 0) {
            std::cerr << "MPP VI: SetChnAttr failed\n"; return false;
        }
        if (VI_EnableDev(0) != 0) { std::cerr << "MPP VI: EnableDev failed\n"; return false; }
        vi_dev_enabled = true;
        if (VI_EnableChn(0, static_cast<VI_CHN>(cfg.vi_chn)) != 0) { std::cerr << "MPP VI: EnableChn failed\n"; return false; }
        vi_chn_enabled = true;
        std::cout << "MPP pipeline: VI/ISP -> NV12 -> BGR\n";
        maybe_init_v2d(cfg.width, cfg.height);
        return true;
    }

    bool read_vi(cv::Mat* out_bgr) {
        if (out_bgr == nullptr || !vi_chn_enabled) return false;
        VideoFrameInfo frame;
        std::memset(&frame, 0, sizeof(frame));
        if (VI_GetChnFrame(0, static_cast<VI_CHN>(cfg.vi_chn), &frame, static_cast<S32>(cfg.timeout_ms)) != 0) {
            return false;
        }
        const bool ok = nv12_to_bgr(frame, out_bgr);
        (void)VI_ReleaseChnFrame(0, static_cast<VI_CHN>(cfg.vi_chn), &frame);
        return ok;
    }

    void close_mpp() {
        release_v2d();
        if (vi_chn_enabled) { (void)VI_DisableChn(0, static_cast<VI_CHN>(cfg.vi_chn)); vi_chn_enabled = false; }
        if (vi_dev_enabled) { (void)VI_DisableDev(0); vi_dev_enabled = false; }
        if (vi_inited) { (void)VI_DeInit(); vi_inited = false; }

        if (vdec_enabled) { (void)VDEC_DisableChn(vdec_chn); vdec_enabled = false; }
        if (vdec_created) { (void)VDEC_DestroyChn(vdec_chn); vdec_created = false; }
        if (vdec_inited) { (void)VDEC_Exit(); vdec_inited = false; }
        if (chn_enabled) { (void)UVC_DisableChn(0, 0); chn_enabled = false; }
        if (dev_enabled) { (void)UVC_DisableDev(0); dev_enabled = false; }
        if (dev_created) { (void)UVC_DestroyDev(0); dev_created = false; }
        if (uvc_inited) { (void)UVC_Exit(); uvc_inited = false; }

        if (vb_inited) { (void)VB_Exit(); vb_inited = false; }
        if (sys_inited) { (void)SYS_Exit(); sys_inited = false; }
        mpp_active = false;
    }
#endif  // VISION_HAS_MPP
};

MppFrameSource::MppFrameSource(const MppFrameSourceConfig& cfg)
    : impl_(std::make_unique<Impl>(cfg)) {}

MppFrameSource::~MppFrameSource() { close(); }

bool MppFrameSource::open() {
    if (!impl_->cfg.use_mpp) {
        return impl_->open_opencv();
    }
#ifdef VISION_HAS_MPP
    const bool ok = impl_->cfg.use_vi ? impl_->open_vi() : impl_->open_uvc();
    if (ok) {
        impl_->mpp_active = true;
        return true;
    }
    // Roll back any partially initialized MPP state (SYS/VB/UVC/VI/VDEC).
    // close_mpp() is idempotent: each step is guarded by its own flag.
    std::cerr << "MppFrameSource: MPP open failed\n";
    impl_->close_mpp();
    return false;
#else
    std::cerr << "MppFrameSource: --use-mpp requested but built without VISION_WITH_MPP; "
            << "falling back to cv::VideoCapture\n";
    return impl_->open_opencv();
#endif
}

bool MppFrameSource::read(cv::Mat* out_bgr) {
    if (out_bgr == nullptr) return false;
#ifdef VISION_HAS_MPP
    if (impl_->mpp_active) {
        return impl_->cfg.use_vi ? impl_->read_vi(out_bgr) : impl_->read_uvc(out_bgr);
    }
#endif
    return impl_->read_opencv(out_bgr);
}

void MppFrameSource::close() {
#ifdef VISION_HAS_MPP
    if (impl_->mpp_active) {
        impl_->close_mpp();
        return;
    }
#endif
    if (impl_->cap_opened) {
        impl_->cap.release();
        impl_->cap_opened = false;
    }
}

}  // namespace vision_mpp
