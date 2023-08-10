#include <opencv2/opencv.hpp>


// Functon for shape detection
void getContours(cv::Mat imgDil, cv::Mat img) {

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(imgDil, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point>> contoursPoly{contours.size()};
    std::vector<cv::Rect> boundRect{contours.size()};
    std::string objectType;

    // Filter out contours that are too small
    for(int i = 0 ; i < contours.size() ; i++) {
        int area = cv::contourArea(contours[i]);
        if(area > 1000) {
            double peri = cv::arcLength(contours[i], true);
            cv::approxPolyDP(contours[i], contoursPoly[i], 0.02*peri, true);
            int objCor = contoursPoly[i].size();
            if(objCor == 3)
                objectType = "Triangle";
            if(objCor == 4)
                objectType = "Rectangle";
            if(objCor > 4)
                objectType = "Circle";
            cv::drawContours(img, contoursPoly, i, cv::Scalar(255,0,255), 2);
            boundRect[i] = cv::boundingRect(contoursPoly[i]);
            cv::rectangle(img, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(0,255,0), 2);
            cv::putText(img, objectType, {boundRect[i].x, boundRect[i].y - 5}, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0,255,0), 1);

        }
    }
}


int main() {

    // // 
    // // Basic OpenCV program - content suggested by ChatGPT when prompted for "OpenCV's most basic program"
    // // 

    // // Load an image from the file system
    // cv::Mat image = cv::imread("C:/Users/pedro/OneDrive/Pictures/14372320_1222593881094194_1371230232141397743_o.jpg");
    // // Check if the image was loaded successfully
    // if (image.empty()) {
    //     std::cout << "Error: Could not open or find the image!" << std::endl;
    //     return -1;
    // }
    // // Display the image in a window
    // cv::imshow("OpenCV Basic Program", image);
    // // Wait for a key press indefinitely (0 means wait forever)
    // cv::waitKey(0);
    // // Close all OpenCV windows
    // cv::destroyAllWindows();



    //
    // Load webcam image and add effects - https://youtu.be/2FYm3GOonhk
    //

    cv::VideoCapture videoSource{0};
    cv::Mat videoFrame;
   
    cv::Mat imgGray, imgBlur, imgEdges, imgDil, imgEro;
    // cv::Mat imgResize, imgCrop;
    // cv::Mat imgCanvas;
    // cv::Mat matrixWarp, imgWarp;
    
    // HSV values and trackbars for color mask
    // cv::Mat imgHSV, imgMask;
    // int hueMin = 0, satMin = 110, valMin = 153;
    // int hueMax = 19, satMax = 240, valMax = 255;
    // cv::namedWindow("Trackbars");
    // cv::createTrackbar("Hue Min", "Trackbars", nullptr, 179);
    // cv::setTrackbarPos("Hue Min", "Trackbars", hueMin);
    // cv::createTrackbar("Hue Max", "Trackbars", nullptr, 179);
    // cv::setTrackbarPos("Hue Max", "Trackbars", hueMax);
    // cv::createTrackbar("Sat Min", "Trackbars", nullptr, 255);
    // cv::setTrackbarPos("Sat Min", "Trackbars", satMin);
    // cv::createTrackbar("Sat Max", "Trackbars", nullptr, 255);
    // cv::setTrackbarPos("Sat Max", "Trackbars", satMax);
    // cv::createTrackbar("Val Min", "Trackbars", nullptr, 255);
    // cv::setTrackbarPos("Val Min", "Trackbars", valMin);
    // cv::createTrackbar("Val Max", "Trackbars", nullptr, 255);
    // cv::setTrackbarPos("Val Max", "Trackbars", valMax);

    // Keep refreshing webcam frame until user quits
    while(true) {

        // Get webcam frame
        videoSource.read(videoFrame);
        // cv::imshow("Webcam", videoFrame);

        // // Effects
        // cv::cvtColor(videoFrame, imgGray, cv::COLOR_BGR2GRAY);
        // cv::GaussianBlur(imgGray, imgBlur, cv::Size(3,3), 3, 0);
        // cv::Canny(imgBlur, imgEdges, 25, 75);
        // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
        // cv::dilate(imgEdges, imgDil, kernel);
        // cv::erode(imgDil, imgEro, kernel);
        // cv::imshow("FX", imgDil);

        // // Resize and crop
        // cv::resize(videoFrame, imgResize, cv::Size(), 1.5, 1.5);
        // imgCrop = videoFrame(cv::Rect(200,200,200,200));
        // cv::imshow("Crop", imgCrop);

        // // Draw shapes and text
        // imgCanvas = cv::Mat(512,512, CV_8UC3, cv::Scalar(255,255,255)); // white canvas
        // cv::circle(videoFrame, cv::Point(256,256), 100, cv::Scalar(255,0,0), 2);
        // cv::rectangle(videoFrame, cv::Point(100,100), cv::Point(200,200), cv::Scalar(0,0,255), 2);
        // cv::line(videoFrame, cv::Point(150,100), cv::Point(250,100), cv::Scalar(0,255,255), 2);
        // cv::putText(videoFrame, "Pedro", cv::Point(250,50), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0,255,0), 1);
        // cv::imshow("Drawing", videoFrame);

        // Warp images
        // cv::Point2f src[4]{{200,200}, {200, 400}, {400, 200}, {400,400}};
        // cv::Point2f dest[4]{{0,0}, {0, 800}, {400, 0}, {400,800}};
        // matrixWarp = cv::getPerspectiveTransform(src, dest);
        // cv::warpPerspective(videoFrame, imgWarp, matrixWarp, cv::Point(400,800));
        // cv::imshow("Warppp", imgWarp);
        // // debug wrap
        // for(auto point : src)
        //     cv::circle(videoFrame, point, 10, cv::Scalar(255,0,0), cv::FILLED);

        // Color detection
        // cv::cvtColor(videoFrame, imgHSV, cv::COLOR_BGR2HSV);
        // cv::imshow("HSV", imgHSV);
        // hueMin = cv::getTrackbarPos("Hue Min", "Trackbars");
        // hueMax = cv::getTrackbarPos("Hue Max", "Trackbars");
        // satMin = cv::getTrackbarPos("Sat Min", "Trackbars");
        // satMax = cv::getTrackbarPos("Sat Max", "Trackbars");
        // valMin = cv::getTrackbarPos("Val Min", "Trackbars");
        // valMax = cv::getTrackbarPos("Val Max", "Trackbars");
        // cv::Scalar lower{(double)hueMin, (double)satMin, (double)valMin};
        // cv::Scalar upper{(double)hueMax, (double)satMax, (double)valMax};
        // cv::inRange(imgHSV, lower, upper, imgMask);
        // cv::imshow("Mask", imgMask);

        // Shape detection
        // cv::cvtColor(videoFrame, imgGray, cv::COLOR_BGR2GRAY);
        // cv::GaussianBlur(imgGray, imgBlur, cv::Size(3,3), 3, 0);
        // cv::Canny(imgBlur, imgEdges, 50, 75);
        // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
        // cv::dilate(imgEdges, imgDil, kernel);
        // getContours(imgDil, videoFrame); // this doesn't really work with webcam because shapes are not well defined

        // Face detection
        cv::CascadeClassifier faceCascade;
        faceCascade.load("resources/haarcascade_frontalface_default.xml");
        if(faceCascade.empty())
            std::cout << "XML file didn't load!" << std::endl;
        std::vector<cv::Rect> faces;
        // face detection works much better on grayscale
        cv::cvtColor(videoFrame, imgGray, cv::COLOR_BGR2GRAY);
        faceCascade.detectMultiScale(imgGray, faces, 1.1, 10);
        // draw rectangle on the detected faces in original webcam image
        for(int i = 0; i < faces.size(); i++) {
            cv::rectangle(videoFrame, faces[i].tl(), faces[i].br(), cv::Scalar(0,0,255), 2);
        }

        // Show original webcam image (with modifications) and wait a bit for next frame
        cv::imshow("Webcam", videoFrame);
        if (cv::waitKey(5) == 27) { // wait for Esc
            break;
        }
    }

    return 0;
}
