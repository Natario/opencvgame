# Coin Catcher

A simple game where you catch coins with your head. Developed with OpenCV and C++

![preview](preview.png "Preview")

# How to run

1. You need to download at least `opencvgame.exe`, the `resources` folder and `opencv_world480.dll` and put them in the same folder.
2. Make sure your webcam is turned on (i.e. not open in another app but available for apps to use it).
3. Run `opencvgame.exe` either by double-clicking or running from the command line.
4. Play! (you need to cover at least 50% of the coin to catch it)

Notes:
1. You can also run `opencvgame.exe hands` from the command line to play a version of the game where you use your hands instead of your head to catch the coins. Please note that this version uses more CPU.
5. If you want to see where the collision boxes are being drawn, run `opencvgame.exe debug`
6. If your head/hands are not detected, try changing the lighting around you so that there is better contrast between your head/hands and the environment.

<br/>

# How to build an OpenCV project in Windows:

1. Download OpenCV for Windows (https://opencv.org/releases/) and install. Also make sure Visual Studio is installed in your system (we're using VS 2022 but other versions should be similar).

    If you're using an IDE (Visual Studio, VS Code, etc.) to write the code (even if you don't use it to compile it), add `C:/opencv/build/include` (or your custom install location) to the include path of the IDE/project so you can use IntelliSense.

2. Create your code file (e.g. `opencvgame.cpp`) and add a basic OpenCV program: 

    ```
    #include <opencv2/opencv.hpp>

    int main() {
        // Load an image from the file system
        cv::Mat image = cv::imread("path/to/your/image.jpg");

        // Check if the image was loaded successfully
        if (image.empty()) {
            std::cout << "Error: Could not open or find the image!" << std::endl;
            return -1;
        }

        // Display the image in a window
        cv::imshow("OpenCV Basic Program", image);

        // Wait for a key press indefinitely (0 means wait forever)
        cv::waitKey(0);

        // Close all OpenCV windows
        cv::destroyAllWindows();

        return 0;
    }
    ```

3. To compile via an IDE like Visual Studio, make sure to add `C:\opencv\build\include` to the include path of the project and `C:\opencv\build\x64\vc16\lib\opencv_world480.lib` to the library path.

4. To compile via the command line, you have to know which command line to use. The OpenCV download probably only had 64bit binaries (in the folder opencv\build there should be only a x64 subfolder and not a x86 one - https://answers.opencv.org/question/229256), so you have to use a 64bit compiler. This means that the `Developer PowerShell for VS 2022` (which comes with Visual Studio) doesn't work because it's 32bit (there isn't a 64bit version as of VS 2022 - https://stackoverflow.com/a/70606900). Instead, open `x64 Native Tools Command Prompt for VS 2022` (which also comes with Visual Studio) and in the command prompt `cd` to the directory of `opencvgame.cpp` and then run:

    ```
    cl /EHsc /I "C:\opencv\build\include" .\opencvgame.cpp /link "C:\opencv\build\x64\vc16\lib\opencv_world480.lib"
    ```

    **Make sure there are no trailing slashes in the include paths (https://stackoverflow.com/a/62404923)**

    Note that building with MingW/g++ doesn't work because the official OpenCV Windows binaries are made for Visual Studio/MSVC. If you want to use MingW/g++, look for custom binaries online or build OpenCV from source.

5. If you try to run the .exe, there might be an error such as:

    ```
    The code execution cannot proceed because opencv_world.dll was not found. Reinstalling the program may fix this problem.
    ```
    You could add `C:\opencv\build\x64\vc16\bin\` to your PATH environment variable, but you can also just copy `C:\opencv\build\x64\vc16\bin\opencv_world480.dll` to the folder where your `opencvgame.exe` is.

6. You should be able to run your executable now.