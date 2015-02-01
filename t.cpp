#include <iostream>
#include <sstream>
#include <iomanip>

#include "opencv/cv.h"
#include "opencv/highgui.h"

#define CVBLOBECLIBAPI extern "C" __declspec(dllexport)
#define CVBLOBLIBAPI __declspec(dllexport)
#include "cvblob.h"
using namespace cvb;

typedef struct key
{
  char  c;
  int x0;
  int y0;
  int x1;
  int y1;
};
key g_keymap[] =
{
	{'2',544,246,579,293},
  {'3',485,247,525,294},
  {'4',438,247,473,294},
  {'5',385,248,425,292},
  {'6',339,251,375,291},
  {'7',287,251,330,291},
  {'8',228,251,273,289},
  {'9',171,251,216,289},
  {'0',113, 248,158, 285},
	{'-',52,240,80,278},
	{'+',10,230,35,270},
	{'Q',558,180,590,214},
  {'W',505,182,545,216},
  {'E',452,185,492,219},
  {'R',396,185,440,220},
  {'T',337,188,383,221},
  {'Y',280,189,326,222},
  {'U',199,188,261,221},
  {'I',167,185,210,219},
  {'O',110,185,156,219},
  {'P', 50,183,100,217},
	{'A',546,137,600,173},
	{'S',482,138,536,174},
  {'D',416,139,472,175},
  {'F',365,141,409,175},
  {'G',313,142,360,175},
  {'H',260,143,309,175},
  {'J',209,141,253,174},
  {'K',155,139,201,174},
  {'L',102, 138,150, 170},
	{';',52,137,92,169},
	{'|',540,95,579,127},
	{'Z',505,96,535,128},
  {'X',462,98,495,130},
  {'C',409,96,452,130},
  {'V',348,98,400,130},
  {'B',292,98,339,130},
  {'N',242,96,282,130},
  {'M',185,93,235,130},
  {'<',136, 93,180,130},
  {'>',80, 93,130, 130},
	{'{',510,53,560,90},
	{'}',442,53,499,90},
  {'_',182,54, 431,90},
	{'"',139,43,179,89},
};
 
int g_key_num = sizeof(g_keymap)/sizeof(key);

int main()
{
 CvTracks tracks;

 cvNamedWindow("red_object_tracking", CV_WINDOW_AUTOSIZE);

 CvCapture *capture = cvCaptureFromCAM(-1);
 cvGrabFrame(capture);
 IplImage *img = cvRetrieveFrame(capture);

 CvSize imgSize = cvGetSize(img);

 IplImage *frame = cvCreateImage(imgSize, img->depth, img->nChannels);

 IplConvKernel* morphKernel = cvCreateStructuringElementEx(5, 5, 1, 1, CV_SHAPE_RECT, NULL);

 //unsigned int frameNumber = 0;
 unsigned int blobNumber = 0;

 bool quit = false;
 while (!quit&&cvGrabFrame(capture))
 {
   IplImage *img = cvRetrieveFrame(capture);

   cvConvertScale(img, frame, 1, 0);

   IplImage *segmentated = cvCreateImage(imgSize, 8, 1);

   // Detecting red pixels:
   // (This is very slow, use direct access better...)
   for (unsigned int j=0; j<imgSize.height; j++)
     for (unsigned int i=0; i<imgSize.width; i++)
     {
    CvScalar c = cvGet2D(frame, j, i);

    double b = ((double)c.val[0])/255.;
    double g = ((double)c.val[1])/255.;
    double r = ((double)c.val[2])/255.;
    // unsigned char f = 255*((r>0.2+g)&&(r>0.2+b));
   // cvSet2D(segmentated, j, i, CV_RGB(f, f, f));
   if(b>0.05 || g>0.05 || r>0.05)
      cvSet2D(segmentated, j, i, CV_RGB(255, 255, 255));
   else
      cvSet2D(segmentated, j, i, CV_RGB(0, 0, 0));
     }

   cvMorphologyEx(segmentated, segmentated, NULL, morphKernel, CV_MOP_OPEN, 1);

   cvShowImage("segmentated", segmentated);

   IplImage *labelImg = cvCreateImage(cvGetSize(frame), IPL_DEPTH_LABEL, 1);

   CvBlobs blobs;
   unsigned int result = cvLabel(segmentated, labelImg, blobs);
   cvFilterByArea(blobs, 500, 1000000);
   cvRenderBlobs(labelImg, blobs, frame, frame, CV_BLOB_RENDER_BOUNDING_BOX);
   cvUpdateTracks(blobs, tracks, 200., 5);
   cvRenderTracks(tracks, frame, frame, CV_TRACK_RENDER_ID|CV_TRACK_RENDER_BOUNDING_BOX);

   cvShowImage("red_object_tracking", frame);

   // print key
   for (CvTracks::const_iterator it=tracks.begin(); it!=tracks.end(); ++it)
   {
      int xx = (int)it->second->centroid.x;
      int yy = (int)it->second->centroid.y;
      //std::cout << xx << ',' << yy << std::endl;

     for(int i=0; i<g_key_num; i++)
     {
        if(xx > g_keymap[i].x0 &&
           xx < g_keymap[i].x1 &&
           yy > g_keymap[i].y0 &&
           yy < g_keymap[i].y1)
        {
            std::cout << g_keymap[i].c << std::endl;
            break;
        }
     }
   }

   cvReleaseImage(&labelImg);
   cvReleaseImage(&segmentated);

   char k = cvWaitKey(06)&0xff;
   switch (k)
   {
     case 27:
     case 'q':
     case 'Q':
       quit = true;
       break;
     case 's':
     case 'S':
       for (CvBlobs::const_iterator it=blobs.begin(); it!=blobs.end(); ++it)
       {
         std::stringstream filename;
         filename << "redobject_blob_" << std::setw(5) << std::setfill('0') << blobNumber << ".png";
         cvSaveImageBlob(filename.str().c_str(), img, it->second);
         blobNumber++;

         std::cout << filename.str() << " saved!" << std::endl;
       }
       break;
   }

   cvReleaseBlobs(blobs);

   //frameNumber++;
 }

 cvReleaseStructuringElement(&morphKernel);
 cvReleaseImage(&frame);

 cvDestroyWindow("red_object_tracking");

 return 0;
}
