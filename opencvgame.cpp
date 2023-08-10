#include <opencv2/opencv.hpp>


void getRandomFigurePos(const cv::Mat img, int& randomX, int& randomY, int figureCols, int figureRows) {
    cv::RNG rng{(uint64)cv::getCPUTickCount()};
    // subtract figure size from max value so that figure doesn't overflow window
    randomX = rng.uniform(0,img.cols - figureCols);
    randomY = rng.uniform(0,img.rows - figureRows);
}

int getRandomFigure(const std::vector<cv::Mat> customFigures) {
    cv::RNG rng{(uint64)cv::getCPUTickCount()};
    return rng.uniform(0, customFigures.size());
}

int main(int argc, char *argv[]) {

    try {

        bool handDetection = false;
        bool debugMode = false;

        // Parse cmd line arguments
        for( int count = 0; count < argc; count++ ) {
            // std::cout << "  argv[" << count << "]   "  << argv[count] << "\n";
            if(strcmp(argv[count], "hand") == 0 || strcmp(argv[count], "hands") == 0)
                handDetection = true;
            if(strcmp(argv[count], "debug") == 0)
                debugMode = true;
        }
        
        cv::VideoCapture videoSource{0};
        cv::Mat videoFrame;
        videoSource.read(videoFrame);     
    
        cv::Mat imgGray, imgClahe, videoFrameMirror;

        int figureX, figureY;
        
        // list of images that will randomly spawn
        std::vector<cv::Mat> customFigures{
            cv::imread("resources/coin1.png", cv::IMREAD_UNCHANGED),
            cv::imread("resources/coin2.png", cv::IMREAD_UNCHANGED),
            cv::imread("resources/coin10.png", cv::IMREAD_UNCHANGED),
            cv::imread("resources/coin20.png", cv::IMREAD_UNCHANGED),
            cv::imread("resources/coin50.png", cv::IMREAD_UNCHANGED)
        };
        // initial random image and position
        int curFigure{getRandomFigure(customFigures)};
        getRandomFigurePos(videoFrame, figureX, figureY, customFigures[curFigure].cols, customFigures[curFigure].rows);

        int score{0};

        // hand detection model from https://github.com/cansik/yolo-hand-detection
        cv::dnn::Net net;
        if(handDetection) {
            net = cv::dnn::readNet("resources/cross-hands-tiny.weights", "resources/cross-hands-tiny.cfg");
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        while(true) {

            // get new video frame
            videoSource.read(videoFrame);
            // mirrored webcam image is easier to know where we should go
            cv::flip(videoFrame, videoFrameMirror, 1);
            
            // face detection works much better on grayscale
            cv::cvtColor(videoFrameMirror, imgGray, cv::COLOR_BGR2GRAY);
            
            // I thought about increasing constrast (and found this https://stackoverflow.com/a/35764666/3174659)
            // to help face detection, but this is only marginally better than using only grayscale
            // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(8,8));
            // clahe->apply(imgGray, imgClahe);
            // cv::imshow("CLAHE", imgClahe);
            
            

            // // 
            // // Use face detection ML model that comes with OpenCV in opencv\sources\data\haarcascades
            // // (poor quality, see better one below)
            // // 
            // cv::CascadeClassifier faceCascade;
            // faceCascade.load("resources/haarcascade_frontalface_default.xml");
            // if(faceCascade.empty())
            //     std::cout << "XML file didn't load!" << std::endl;
            // std::vector<cv::Rect> faces;
            // faceCascade.detectMultiScale(imgGray, faces, 1.1, 10);

            // for(int i = 0; i < faces.size(); i++) {
            //     // debug rectangle
            //     cv::rectangle(videoFrameMirror, faces[i].tl(), faces[i].br(), cv::Scalar(0,255,0), 2);

            //     // calculate if there is overlap between face and figure - https://answers.opencv.org/question/59544/percentage-of-overlap/
            //     int left = cv::max(faces[i].tl().x, figureX);
            //     int right = cv::min(faces[i].br().x, figureX+customFigures[curFigure].cols);
            //     int bottom = cv::min(faces[i].br().y, figureY+customFigures[curFigure].rows);
            //     int top = cv::max(faces[i].tl().y, figureY);
            //     // there's overlap, so it's a hit!
            //     if(left < right && bottom > top) {
            //         // BUT mark as a hit only if a substantial amount of the face is on top of the figure
            //         double intersectionArea = (right - left) * (bottom - top);
            //         double figureArea = customFigures[curFigure].cols * customFigures[curFigure].rows;
            //         double percentageIntersect = intersectionArea / figureArea;
            //         // std::cout << percentageIntersect << std::endl;
            //         if(percentageIntersect > 0.5) {
            //             getRandomFigurePos(videoFrameMirror, figureX, figureY, customFigures[curFigure].cols, customFigures[curFigure].rows);
            //             curFigure = getRandomFigure(customFigures);
            //             score++;
            //         }
            //     }
            // }


            // 
            // This is a much better face detection algorithm! https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html (the important code is at the bottom)
            // thanks to a suggestion from https://pyimagesearch.com/2021/04/26/face-detection-tips-suggestions-and-best-practices/
            // 
            if(!handDetection) {
               
                cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(
                    "resources/face_detection_yunet_2023mar.onnx",
                    "",
                    cv::Size(videoFrameMirror.cols, videoFrameMirror.rows)
                );
                cv::Mat faces;
                detector->detect(videoFrameMirror, faces);
                
                for(int i = 0; i < faces.rows; i++) {
                    // debug rectangle
                    if(debugMode)
                        cv::rectangle(videoFrameMirror,cv::Rect2i(int(faces.at<float>(i, 0)),int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), cv::Scalar(0, 255, 0), 2);

                // calculate if there is overlap between face and figure - https://answers.opencv.org/question/59544/percentage-of-overlap/
                    int left = cv::max(int(faces.at<float>(i, 0)), figureX);
                    int right = cv::min(int(faces.at<float>(i, 0) + faces.at<float>(i, 2)), figureX+customFigures[curFigure].cols);
                    int bottom = cv::min(int(faces.at<float>(i, 1) + faces.at<float>(i, 3)), figureY+customFigures[curFigure].rows);
                    int top = cv::max(int(faces.at<float>(i, 1)), figureY);
                    // there's overlap, so it's a hit!
                    if(left < right && bottom > top) {
                        // BUT mark as a hit only if a substantial amount of the face is on top of the figure
                        double intersectionArea = (right - left) * (bottom - top);
                        double figureArea = customFigures[curFigure].cols * customFigures[curFigure].rows;
                        double percentageIntersect = intersectionArea / figureArea;
                        // std::cout << percentageIntersect << std::endl;
                        if(percentageIntersect > 0.5) {
                            getRandomFigurePos(videoFrameMirror, figureX, figureY, customFigures[curFigure].cols, customFigures[curFigure].rows);
                            curFigure = getRandomFigure(customFigures);
                            score++;
                        }
                    }
                }
            
            
            } else {


                //
                // Hand detection algorithm suggested by ChatGPT
                //

                // Preprocess input image
                cv::Mat blob = cv::dnn::blobFromImage(videoFrameMirror, 0.00392, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);

                // Perform object detection
                net.setInput(blob);
                std::vector<cv::Mat> outs;
                net.forward(outs, net.getUnconnectedOutLayersNames());
                
                // Post-processing and visualization
                std::vector<int> classIds;
                std::vector<float> confidences;
                std::vector<cv::Rect> boxes;

                for (const cv::Mat& out : outs) {
                    for (int i = 0; i < out.rows; ++i) {
                        const float* data = out.ptr<float>(i);
                        for (int j = 0; j < out.cols; ++j, data += 85) { // 85 values per detection
                            cv::Mat scores = out.row(i).colRange(5, out.cols);
                            cv::Point classIdPoint;
                            double confidence;

                            // Get class ID with maximum confidence
                            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                            if (confidence > 0.5) { // Confidence threshold
                                int centerX = static_cast<int>(data[0] * videoFrameMirror.cols);
                                int centerY = static_cast<int>(data[1] * videoFrameMirror.rows);
                                int width = static_cast<int>(data[2] * videoFrameMirror.cols);
                                int height = static_cast<int>(data[3] * videoFrameMirror.rows);
                                int left = centerX - width / 2;
                                int top = centerY - height / 2;

                                classIds.push_back(classIdPoint.x);
                                confidences.push_back(static_cast<float>(confidence));
                                boxes.push_back(cv::Rect(left, top, width, height));
                                
                                // std::cout << "i: " << i << std::endl;
                                // std::cout << "j: " << j << std::endl;
                                // std::cout << "confidence: " << confidence << std::endl;
                                // std::cout << "left: " << left << std::endl;
                                // std::cout << "top: " << top << std::endl;
                                // std::cout << "-------------" << std::endl;
                            }

                            break; // only the first j value seems to be the real hand detection. the rest detections seem to be bogus (they're probably other kinds of data)
                        }
                    }
                }

                // Non-maximum suppression to remove duplicate detections
                std::vector<int> indices;
                cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

                for (size_t i = 0; i < indices.size(); ++i) {
                    int idx = indices[i];
                    cv::Rect box = boxes[idx];
                    // debug rectangle
                    if(debugMode)
                        cv::rectangle(videoFrameMirror, box, cv::Scalar(0, 255, 0), 2);

                    // calculate if there is overlap between face and figure - https://answers.opencv.org/question/59544/percentage-of-overlap/
                    int left = cv::max(box.tl().x, figureX);
                    int right = cv::min(box.br().x, figureX+customFigures[curFigure].cols);
                    int bottom = cv::min(box.br().y, figureY+customFigures[curFigure].rows);
                    int top = cv::max(box.tl().y, figureY);
                    // there's overlap, so it's a hit!
                    if(left < right && bottom > top) {
                        // BUT mark as a hit only if a substantial amount of the face is on top of the figure
                        double intersectionArea = (right - left) * (bottom - top);
                        double figureArea = customFigures[curFigure].cols * customFigures[curFigure].rows;
                        double percentageIntersect = intersectionArea / figureArea;
                        // std::cout << percentageIntersect << std::endl;
                        if(percentageIntersect > 0.5) {
                            getRandomFigurePos(videoFrameMirror, figureX, figureY, customFigures[curFigure].cols, customFigures[curFigure].rows);
                            curFigure = getRandomFigure(customFigures);
                            score++;
                        }
                    }
                }
            
            
            } // end if(faceDetection)

            

            // Draw figure to hit - https://stackoverflow.com/a/29398615
            cv::Mat insetImage{videoFrameMirror, cv::Rect(figureX, figureY, customFigures[curFigure].cols, customFigures[curFigure].rows)};
            // overlay figure basic
            // customFigures[curFigure].copyTo(insetImage);
            // overlay figure with alpha channel! https://stackoverflow.com/a/32481105 via a comment to https://stackoverflow.com/q/40895785
            double alpha = 1; // alpha in [0,1]
            for (int r = 0; r < insetImage.rows; ++r) {
                for (int c = 0; c < insetImage.cols; ++c) {
                    const cv::Vec4b& vf = customFigures[curFigure].at<cv::Vec4b>(r,c);
                    if (vf[3] > 0) { // alpha channel > 0
                        // Blending
                        cv::Vec3b& vb = insetImage.at<cv::Vec3b>(r,c);
                        vb[0] = alpha * vf[0] + (1 - alpha) * vb[0];
                        vb[1] = alpha * vf[1] + (1 - alpha) * vb[1];
                        vb[2] = alpha * vf[2] + (1 - alpha) * vb[2];
                    }
                }
            }
            // rectangle that can be used as a figure to hit or as a debug bounding box of custom image
            if(debugMode)
                cv::rectangle(videoFrameMirror, {figureX,figureY} , {figureX+customFigures[curFigure].cols,figureY+customFigures[curFigure].rows}, cv::Scalar(0,0,255), 2);
            

            

            // Score
            cv::putText(videoFrameMirror, "You have caught " +  std::to_string(score) + " coins!", cv::Point(videoFrameMirror.cols/2-180,30), cv::FONT_HERSHEY_COMPLEX, .75, cv::Scalar(0,255,0), 1);
            
            cv::imshow("Webcam", videoFrameMirror);
            if (cv::waitKey(20) == 27) { // Esc leaves game
                break;
            }
        }

        return 0;

    } catch (const std::exception &exc) {
        std::cerr << exc.what();
    }
}