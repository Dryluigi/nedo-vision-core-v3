#include <nvdsmeta.h>
#include <nvdsinfer.h>
#include <nvbufsurface.h>
#include <opencv2/opencv.hpp>

static cv::Mat helmet_icon;
static cv::Mat vest_icon;

static bool initialized = false;

static void load_icons()
{
    if (initialized) return;

    helmet_icon = cv::imread("/app/icons/helmet.png", cv::IMREAD_UNCHANGED);
    vest_icon   = cv::imread("/app/icons/vest.png", cv::IMREAD_UNCHANGED);

    cv::resize(helmet_icon, helmet_icon, cv::Size(40, 40));
    cv::resize(vest_icon, vest_icon, cv::Size(40, 40));

    initialized = true;
}

static float IoU(NvOSD_RectParams a, NvOSD_RectParams b)
{
    float x1 = std::max(a.left, b.left);
    float y1 = std::max(a.top, b.top);
    float x2 = std::min(a.left + a.width, b.left + b.width);
    float y2 = std::min(a.top + a.height, b.top + b.height);

    float interArea = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    float unionArea = a.width * a.height + b.width * b.height - interArea;

    if (unionArea <= 0) return 0.0f;

    return interArea / unionArea;
}

extern "C" gboolean NvDsCudaProcess(NvBufSurface *surface,
                                    NvDsBatchMeta *batch_meta,
                                    gpointer user_data)
{
    load_icons();

    NvBufSurfaceMap(surface, -1, -1, NVBUF_MAP_READ_WRITE);

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
         l_frame != NULL;
         l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

        cv::Mat frame(surface->surfaceList[0].height,
                      surface->surfaceList[0].width,
                      CV_8UC4,
                      surface->surfaceList[0].mappedAddr.addr[0],
                      surface->surfaceList[0].pitch);

        std::vector<NvDsObjectMeta*> persons;
        std::vector<NvDsObjectMeta*> helmets;
        std::vector<NvDsObjectMeta*> vests;

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list;
             l_obj != NULL;
             l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)l_obj->data;

            if (obj->class_id == 0) persons.push_back(obj);
            if (obj->class_id == 1) helmets.push_back(obj);
            if (obj->class_id == 2) vests.push_back(obj);
        }

        for (auto person : persons)
        {
            int icon_offset_y = 0;

            for (auto helmet : helmets)
            {
                if (IoU(person->rect_params, helmet->rect_params) > 0.1)
                {
                    cv::Rect roi(person->rect_params.left,
                                 person->rect_params.top - 45 - icon_offset_y,
                                 40, 40);

                    helmet_icon.copyTo(frame(roi));
                    icon_offset_y += 45;
                }
            }

            for (auto vest : vests)
            {
                if (IoU(person->rect_params, vest->rect_params) > 0.1)
                {
                    cv::Rect roi(person->rect_params.left,
                                 person->rect_params.top - 45 - icon_offset_y,
                                 40, 40);

                    vest_icon.copyTo(frame(roi));
                    icon_offset_y += 45;
                }
            }
        }
    }

    NvBufSurfaceUnMap(surface, -1, -1);

    return TRUE;
}