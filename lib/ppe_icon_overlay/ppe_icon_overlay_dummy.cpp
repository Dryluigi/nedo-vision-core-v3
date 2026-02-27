#include <nvdsmeta.h>
#include <nvbufsurface.h>
#include <opencv2/opencv.hpp>

static cv::Mat helmet_icon;
static cv::Mat vest_icon;
static bool initialized = false;

static void load_icons()
{
    if (initialized) return;

    helmet_icon = cv::imread("/app/assets/icons/helmet-green.png", cv::IMREAD_UNCHANGED);
    vest_icon   = cv::imread("/app/assets/icons/vest-green.png", cv::IMREAD_UNCHANGED);

    if (helmet_icon.empty() || vest_icon.empty()) {
        printf("❌ Failed to load icons\n");
        return;
    }

    cv::resize(helmet_icon, helmet_icon, cv::Size(60, 60));
    cv::resize(vest_icon, vest_icon, cv::Size(60, 60));

    initialized = true;
    printf("✅ Icons loaded successfully\n");
}

extern "C" gboolean NvDsCudaProcess(NvBufSurface *surface,
                                    NvDsBatchMeta *batch_meta,
                                    gpointer user_data)
{
    load_icons();

    if (!initialized) return TRUE;

    NvBufSurfaceMap(surface, -1, -1, NVBUF_MAP_READ_WRITE);

    for (guint i = 0; i < surface->batchSize; i++)
    {
        cv::Mat frame(surface->surfaceList[i].height,
                      surface->surfaceList[i].width,
                      CV_8UC4,
                      surface->surfaceList[i].mappedAddr.addr[0],
                      surface->surfaceList[i].pitch);

        int x = 50;
        int y = 50;

        // Draw helmet
        if (!helmet_icon.empty()) {
            cv::Rect roi1(x, y, helmet_icon.cols, helmet_icon.rows);
            helmet_icon.copyTo(frame(roi1));
        }

        // Draw vest below helmet
        if (!vest_icon.empty()) {
            cv::Rect roi2(x, y + 70, vest_icon.cols, vest_icon.rows);
            vest_icon.copyTo(frame(roi2));
        }
    }

    NvBufSurfaceUnMap(surface, -1, -1);

    return TRUE;
}