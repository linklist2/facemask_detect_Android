package com.example.facemask_detect;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "yolov5";
    private CameraBridgeViewBase mOpenCvCameraView;
    private Net net;
    private static final double[][] colorList = {{56, 56, 255}, {151, 157, 255}};
    private static final String[] classNames = {
            "face",
            "face_mask"
    };


    // Initialize OpenCV manager.
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    @Override
    protected void onStart() {
        super.onStart();
        // 获取摄像头权限
        Permission.checkPermission(this);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // 设置app全屏状态
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);


        // Set up camera listener.
        mOpenCvCameraView = findViewById(R.id.CameraView);
        // grant, 需要加上这句话否则是黑屏状态，无法调用摄像头
        mOpenCvCameraView.setCameraPermissionGranted();
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        // 如果授权摄像头才会加载opencv
        if (Permission.isPermissionGranted(this)) {
            // 加载动态库
            if (OpenCVLoader.initDebug()) {
                Log.d("OpenCV", "OpenCV library found inside package. Using it!");
                mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            }
            // 加载动态库失败，就使用opencv Manager进行初始化
            else {
                Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
                OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
            }
        }


    }


    private String getPath(String file, Context context) {
        // 获得文件的绝对路径
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }


    public void letterbox(@NonNull Mat frame, @NonNull Mat out, int width, int height) {
        // 具体解释可以看我的博客：https://blog.csdn.net/weixin_43490422/article/details/127148825?spm=1001.2014.3001.5501
        // height, width
        Size size = frame.size();
        Double ratio = Double.min(width / size.width, height / size.height);

        int width_pad = (int) Math.round(size.width * ratio);
        int height_pad = (int) Math.round(size.height * ratio);

        int dw = width - width_pad;
        int dh = height - height_pad;

        double dw_double = dw / 2.0;
        double dh_double = dh / 2.0;

        Size new_size = new Size(width_pad, height_pad);

        // 先将最长边缩放至指定的尺寸
        if (size.height != height_pad || size.width != width_pad) {
            Imgproc.resize(frame, out, new_size);
        }

        int top = (int) (Math.round(dh_double - 0.1));
        int bottom = (int) (Math.round(dh_double + 0.1));
        int left = (int) (Math.round(dw_double - 0.1));
        int right = (int) (Math.round(dw_double + 0.1));

        // 用灰色填充没有达到size的地方
        Core.copyMakeBorder(out, out, top, bottom, left, right, Core.BORDER_CONSTANT, new Scalar(114, 114, 114));

    }

    public int clamp(int value, int min, int max) {
        // 将value限制到[min,max]区间内
        if (value > max) {
            value = max;

        } else if (value < min) {
            value = min;
        }

        return value;

    }

    private void draw(Integer class_id, Float confidence, int top_x, int top_y, int bottom_x, int bottom_y, Mat frame) {
        // 绘制目标矩形框
        int thickness = Integer.max((int) ((frame.cols() + frame.rows()) / 2 * 0.003), 2);
        Imgproc.rectangle(frame, new Point(top_x, top_y), new Point(bottom_x, bottom_y),
                new Scalar(colorList[class_id]), thickness);

        // 目标框标签，只保留2位小数
        DecimalFormat fnum = new DecimalFormat("##0.00");
        String label = classNames[class_id] + ": " + fnum.format(confidence);
        int tf = Integer.max(thickness - 1, 1);
        Size textSize = Imgproc.getTextSize(label, 0, (int) (thickness / 3), tf, null);
        boolean outsize = top_y - textSize.height >= 3;
        if (outsize) {
            Imgproc.rectangle(frame, new Point(top_x, top_y),
                    new Point(top_x + textSize.width, top_y - textSize.height - 3),
                    new Scalar(colorList[class_id]), -1);

            Imgproc.putText(frame, label,
                    new Point(top_x, top_y - 2),
                    0,
                    thickness / 3,
                    new Scalar(255, 255, 255),
                    tf);
        } else {
            Imgproc.rectangle(frame, new Point(top_x, top_y),
                    new Point(top_x + textSize.width, top_y + textSize.height + 3),
                    new Scalar(colorList[class_id]), -1);

            Imgproc.putText(frame, label,
                    new Point(top_x, top_y + textSize.height + 2),
                    0,
                    thickness / 3,
                    new Scalar(255, 255, 255),
                    tf);
        }

    }


    public void post_process(@NonNull Mat frame, @NonNull Mat out, @NonNull Mat dst, double conf_threshold, double iou_threshold, int max_wh) {
        List<Integer> classIds = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        List<Rect2d> boxes = new ArrayList<>();


        for (int i = 0; i < out.rows(); i++) {
            // 得到目标置信度，也就是框中存在目标的概率
            double confidence = out.get(i, 4)[0];
            if (confidence > conf_threshold) {
                // 得到预测的类别号
                Mat scores = out.row(i).colRange(5, out.cols());
                Core.MinMaxLocResult result = Core.minMaxLoc(scores);
                int cls_id = (int) result.maxLoc.x;
                classIds.add(cls_id);

                // 将当前置信度 * 最大分类概率作为新的置信度,再传入后面的NMSBoxes.
                confidences.add((float) (confidence * result.maxVal));


                //将（中心点x,中心点y，宽度，高度）转成cv2.dnn.NMSBoxes所需要的（左上角x，左上角y，宽度，高度）

                // opencv中比较特殊，横轴为height，纵轴为width; 横轴为y，纵轴为x;
                int x_center = (int) out.get(i, 0)[0];
                int y_center = (int) out.get(i, 1)[0];
                int width = (int) out.get(i, 2)[0];
                int height = (int) out.get(i, 3)[0];
                // + cls_id * max_wh 是为了进行offset，以便区分开不同类别的预测框
                int top_x = (x_center - width / 2) + cls_id * max_wh;
                int top_y = (y_center - height / 2) + cls_id * max_wh;
                boxes.add(new Rect2d(top_x, top_y, width, height));

            }

        }
        MatOfRect2d boxs = new MatOfRect2d();
        boxs.fromList(boxes);
        MatOfFloat confis = new MatOfFloat();
        confis.fromList(confidences);
        MatOfInt idx_mat = new MatOfInt();

        // 非极大值抑制，注意:删选过后的box是乘上偏移的
        Dnn.NMSBoxes(boxs, confis, (float) conf_threshold, (float) iou_threshold, idx_mat);

        // 如果筛选过后，还有预测框
        if (idx_mat.total() > 0) {
            int[] indices = idx_mat.toArray();
            float gain = Float.min((float) dst.rows() / (float) frame.rows(), (float) dst.cols() / (float) frame.cols());
            int x_pad = (int) (dst.cols() - frame.cols() * gain) / 2;
            int y_pad = (int) (dst.rows() - frame.rows() * gain) / 2;


            for (int i = 0; i < indices.length; ++i) {
                int box_idx = indices[i]; // box的索引
                Rect2d box = boxes.get(box_idx);
                int cls_id = classIds.get(box_idx); // 类索引

                // 得到还原后的预测框。首先要将偏移恢复；然后再把填充恢复，最后还原回原来的比例；
                int top_x = (int) (((box.x - cls_id * max_wh) - x_pad) / gain);
                int top_y = (int) (((box.y - cls_id * max_wh) - y_pad) / gain);
                int bottom_x = (int) (top_x + box.width / gain);
                int bottom_y = (int) (top_y + box.height / gain);

                // 为了预测框不超出边界，做一下限制
                top_x = clamp(top_x, 0, frame.cols());
                bottom_x = clamp(bottom_x, 0, frame.cols());
                top_y = clamp(top_y, 0, frame.rows());
                bottom_y = clamp(bottom_y, 0, frame.rows());


                // 绘制目标框
                draw(classIds.get(box_idx), confidences.get(box_idx), top_x, top_y, bottom_x, bottom_y, frame);
            }
        }


    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        // 加载模型
        net = Dnn.readNetFromONNX(getPath("yolov5n_mask.onnx", this));
        Log.i(TAG, "Network loaded successfully");
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        // resize parameters
        final int IN_WIDTH = 320;
        final int IN_HEIGHT = 320;

        // NMS parameters
        final double conf_thres = 0.40;
        final double iou_thres = 0.45;
        final int max_wh = 4096; // offset value

        // 图像预处理, frame的数据类型是CV_8UC3表示：8位无符号整数，通道数为3
        Mat frame = inputFrame.rgba();

        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
        Mat dst = new Mat();
        letterbox(frame, dst, IN_WIDTH, IN_HEIGHT);

        // 推理
        Mat blob = Dnn.blobFromImage(dst, 1 / 255.0);
        net.setInput(blob);
        Mat detections = net.forward(); //detections的数据类型32FC1:32位float，通道1

        // 后处理及绘制目标框
        detections = detections.reshape(0, (int) detections.size().width);
        post_process(frame, detections, dst, conf_thres, iou_thres, max_wh);

        return frame;
    }
}