#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <VapourSynth.h>
#include <VSHelper.h>


// The largest is 15x15, but the center pixel gets added three times.
// + 1 because 227 needs to be a valid index in the magic array.
#define MAX_PIXEL_COUNT 227 + 1


template <typename PixelType>
static void process_plane_scalar(const uint8_t *srcp8, uint8_t *dstp8, int first_column, int width, int height, int stride, unsigned threshold, int radius, const unsigned magic[MAX_PIXEL_COUNT]) {
    (void)magic;

    const PixelType *srcp = (const PixelType *)srcp8;
    PixelType *dstp = (PixelType *)dstp8;
    stride /= sizeof(PixelType);

    for (int y = 0; y < height; y++) {
        for (int x = first_column; x < width; x++) {
            unsigned center_pixel = srcp[x];

            unsigned sum = center_pixel * 2;
            unsigned counter = 2;

            for (int yy = std::max(-y, -radius); yy <= std::min(radius, height - y - 1); yy++) {
                for (int xx = std::max(-x, -radius); xx <= std::min(radius, width - x - 1); xx++) {
                    unsigned neighbour_pixel = srcp[x + yy * stride + xx];

                    if (threshold > (unsigned)std::abs((int)center_pixel - (int)neighbour_pixel)) {
                        counter++;
                        sum += neighbour_pixel;
                    }
                }
            }

            dstp[x] = (sum * 2 + counter) / (counter * 2);
        }

        srcp += stride;
        dstp += stride;
    }
}


#if defined (MINIDEEN_X86)

#include <emmintrin.h>


#define zeroes _mm_setzero_si128()


static void process_plane_sse2_8bit(const uint8_t *srcp, uint8_t *dstp, int first_column, int width, int height, int stride, unsigned threshold, int radius, const unsigned magic[MAX_PIXEL_COUNT]) {
    (void)first_column; // Always 0 in this function.

    const uint8_t *srcp_orig = srcp;
    uint8_t *dstp_orig = dstp;

    __m128i th = _mm_set1_epi16(threshold);

    int width_16 = width / 16 * 16;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width_16; x += 16) {
            __m128i center_pixel = _mm_loadu_si128((const __m128i *)&srcp[x]);

            __m128i center_lo = _mm_unpacklo_epi8(center_pixel, zeroes);
            __m128i center_hi = _mm_unpackhi_epi8(center_pixel, zeroes);

            __m128i sum_lo = _mm_slli_epi16(center_lo, 1);
            __m128i sum_hi = _mm_slli_epi16(center_hi, 1);

            __m128i counter_lo = _mm_set1_epi16(2);
            __m128i counter_hi = counter_lo;

            for (int yy = std::max(-y, -radius); yy <= std::min(radius, height - y - 1); yy++) {
                for (int xx = std::max(-x, -radius); xx <= std::min(radius, width - x - 16); xx++) {
                    __m128i neighbour_pixel = _mm_loadu_si128((const __m128i *)&srcp[x + yy * stride + xx]);

                    __m128i neighbour_lo = _mm_unpacklo_epi8(neighbour_pixel, zeroes);
                    __m128i neighbour_hi = _mm_unpackhi_epi8(neighbour_pixel, zeroes);

                    __m128i abs_lo = _mm_or_si128(_mm_subs_epu16(center_lo, neighbour_lo),
                                                  _mm_subs_epu16(neighbour_lo, center_lo));
                    __m128i abs_hi = _mm_or_si128(_mm_subs_epu16(center_hi, neighbour_hi),
                                                  _mm_subs_epu16(neighbour_hi, center_hi));

                    __m128i gt_mask_lo = _mm_cmpgt_epi16(th, abs_lo);
                    __m128i gt_mask_hi = _mm_cmpgt_epi16(th, abs_hi);

                    // Subtract 65535 aka -1
                    counter_lo = _mm_sub_epi16(counter_lo, gt_mask_lo);
                    counter_hi = _mm_sub_epi16(counter_hi, gt_mask_hi);

                    __m128i gt_pixels_lo = _mm_and_si128(gt_mask_lo, neighbour_lo);
                    __m128i gt_pixels_hi = _mm_and_si128(gt_mask_hi, neighbour_hi);

                    sum_lo = _mm_adds_epu16(sum_lo, gt_pixels_lo);
                    sum_hi = _mm_adds_epu16(sum_hi, gt_pixels_hi);
                }
            }

            sum_lo = _mm_add_epi16(sum_lo,
                                   _mm_srli_epi16(counter_lo, 1));
            sum_hi = _mm_add_epi16(sum_hi,
                                   _mm_srli_epi16(counter_hi, 1));

            __m128i magic_lo = zeroes;
            __m128i magic_hi = zeroes;

            for (int i = 0; i < 8; i++) {
                uint16_t e_lo = _mm_extract_epi16(counter_lo, i);
                uint16_t e_hi = _mm_extract_epi16(counter_hi, i);
                magic_lo = _mm_insert_epi16(magic_lo, magic[e_lo], i);
                magic_hi = _mm_insert_epi16(magic_hi, magic[e_hi], i);
            }

            __m128i result_lo = _mm_mulhi_epu16(sum_lo, magic_lo);
            __m128i result_hi = _mm_mulhi_epu16(sum_hi, magic_hi);

            _mm_storeu_si128((__m128i *)&dstp[x],
                             _mm_packus_epi16(result_lo, result_hi));
        }
        /// sum / counter
        /// there is a number which, when multiplied by counter, results in 65536 (approximately)
        /// so multiply sum by the same number, and shift the result right 16 bits
        /// magic number: 65536 / counter
        ///
        /// (sum * magic + 32768) >> 16
        ///
        /// (sum * (65536 / counter) + 32768) >> 16
        /// (sum * 65536 / counter + 32768) >> 16
        /// (sum * 65536 / counter + 32768 * counter / counter) >> 16
        /// ((sum * 65536 + counter * 32768) / counter) >> 16
        /// ((sum * 2 * 32768 + counter * 32768) / counter) >> 16
        /// ((sum * 2 + counter) * 32768 / counter) >> 16
        /// ((sum * 2 + half_counter * 2) * 32768 / counter) >> 16
        /// (sum + half_counter) * 65536 / counter) >> 16

        srcp += stride;
        dstp += stride;
    }

    if (width_16 < width) {
        process_plane_scalar<uint8_t>(srcp_orig,
                                      dstp_orig,
                                      width_16,
                                      width,
                                      height,
                                      stride,
                                      threshold,
                                      radius,
                                      magic);
    }
}

#endif // MINIDEEN_X86


typedef struct MiniDeenData {
    VSNodeRef *clip;
    const VSVideoInfo *vi;

    int radius[3];
    int threshold[3];
    int process[3];

    unsigned magic[MAX_PIXEL_COUNT];

    decltype (process_plane_scalar<uint8_t>) *process_plane;
} MiniDeenData;


static void VS_CC minideenInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    (void)in;
    (void)out;
    (void)core;

    MiniDeenData *d = (MiniDeenData *) *instanceData;

    vsapi->setVideoInfo(d->vi, 1, node);
}


static const VSFrameRef *VS_CC minideenGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    (void)frameData;

    const MiniDeenData *d = (const MiniDeenData *) *instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->clip, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->clip, frameCtx);

        const VSFrameRef *plane_src[3] = {
            d->process[0] ? nullptr : src,
            d->process[1] ? nullptr : src,
            d->process[2] ? nullptr : src
        };
        int planes[3] = { 0, 1, 2 };

        VSFrameRef *dst = vsapi->newVideoFrame2(d->vi->format,
                                                vsapi->getFrameWidth(src, 0),
                                                vsapi->getFrameHeight(src, 0),
                                                plane_src,
                                                planes,
                                                src,
                                                core);

        for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
            if (!d->process[plane])
                continue;

            const uint8_t *srcp = vsapi->getReadPtr(src, plane);
            uint8_t *dstp = vsapi->getWritePtr(dst, plane);
            int stride = vsapi->getStride(src, plane);
            int width = vsapi->getFrameWidth(src, plane);
            int height = vsapi->getFrameHeight(src, plane);

            d->process_plane(srcp, dstp, 0, width, height, stride, d->threshold[plane], d->radius[plane], d->magic);
        }

        vsapi->freeFrame(src);

        return dst;
    }

    return nullptr;
}


static void VS_CC minideenFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    (void)core;

    MiniDeenData *d = (MiniDeenData *)instanceData;

    vsapi->freeNode(d->clip);
    free(d);
}


static void VS_CC minideenCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    (void)userData;

    MiniDeenData d;
    memset(&d, 0, sizeof(d));

    int err;

    for (int i = 0; i < 3; i++) {
        d.radius[i] = int64ToIntS(vsapi->propGetInt(in, "radius", i, &err));
        if (err)
            d.radius[i] = (i == 0) ? 1
                                   : d.radius[i - 1];

        d.threshold[i] = int64ToIntS(vsapi->propGetInt(in, "threshold", i, &err));
        if (err)
            d.threshold[i] = (i == 0) ? 10
                                      : d.threshold[i - 1];

        if (d.radius[i] < 1 || d.radius[i] > 7) {
            vsapi->setError(out, "MiniDeen: radius must be between 1 and 7 (inclusive).");
            return;
        }

        if (d.threshold[i] < 0 || d.threshold[i] > 255) {
            vsapi->setError(out, "MiniDeen: threshold must be between 0 and 255 (inclusive).");
            return;
        }
    }

    d.clip = vsapi->propGetNode(in, "clip", 0, nullptr);
    d.vi = vsapi->getVideoInfo(d.clip);

    if (!d.vi->format ||
        d.vi->format->bitsPerSample > 16 ||
        d.vi->format->sampleType != stInteger) {
        vsapi->setError(out, "MiniDeen: only 8..16 bit integer clips with constant format are supported.");
        vsapi->freeNode(d.clip);
        return;
    }


    int n = d.vi->format->numPlanes;
    int m = vsapi->propNumElements(in, "planes");

    for (int i = 0; i < 3; i++)
        d.process[i] = (m <= 0);

    for (int i = 0; i < m; i++) {
        int o = int64ToIntS(vsapi->propGetInt(in, "planes", i, 0));

        if (o < 0 || o >= n) {
            vsapi->freeNode(d.clip);
            vsapi->setError(out, "MiniDeen: plane index out of range");
            return;
        }

        if (d.process[o]) {
            vsapi->freeNode(d.clip);
            vsapi->setError(out, "MiniDeen: plane specified twice");
            return;
        }

        d.process[o] = 1;
    }


    int pixel_max = (1 << d.vi->format->bitsPerSample) - 1;

    for (int i = 0; i < 3; i++)
        d.threshold[i] = d.threshold[i] * pixel_max / 255;


    int opt = vsapi->propGetInt(in, "opt", 0, &err);

    if (opt == 0) {
        d.process_plane = (d.vi->format->bitsPerSample == 8) ? process_plane_scalar<uint8_t>
                                                             : process_plane_scalar<uint16_t>;
    } else if (opt == 1) {
        d.process_plane = process_plane_sse2_8bit;
    }



    for (int i = 1; i < MAX_PIXEL_COUNT; i++)
        d.magic[i] = (unsigned)(65536.0 / i + 0.5);


    MiniDeenData *data = (MiniDeenData *)malloc(sizeof(d));
    *data = d;

    vsapi->createFilter(in, out, "MiniDeen", minideenInit, minideenGetFrame, minideenFree, fmParallel, 0, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.nodame.minideen", "minideen", "Spatial convolution with thresholds", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("MiniDeen",
                 "clip:clip;"
                 "radius:int[]:opt;"
                 "threshold:int[]:opt;"
                 "planes:int[]:opt;"
                 "opt:int:opt;"
                 , minideenCreate, 0, plugin);
}
