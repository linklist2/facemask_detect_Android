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
        // ?????????????????????
        Permission.checkPermission(this);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // ??????app????????????
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);


        // Set up camera listener.
        mOpenCvCameraView = findViewById(R.id.CameraView);
        // grant, ??????????????????????????????????????????????????????????????????
        mOpenCvCameraView.setCameraPermissionGranted();
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        // ?????????????????????????????????opencv
        if (Permission.isPermissionGranted(this)) {
            // ???????????????
            if (OpenCVLoader.initDebug()) {
                Log.d("OpenCV", "OpenCV library found inside package. Using it!");
                mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            }
            // ?????????????????????????????????opencv Manager???????????????
            else {
                Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
                OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
            }
        }


    }


    private String getPath(String file, Context context) {
        // ???????????????????????????
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
        // ????????????????????????????????????https://blog.csdn.net/weixin_43490422/article/details/127148825?spm=1001.2014.3001.5501
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

        // ???????????????????????????????????????
        if (size.height != height_pad || size.width != width_pad) {
            Imgproc.resize(frame, out, new_size);
        }

        int top = (int) (Math.round(dh_double - 0.1));
        int bottom = (int) (Math.round(dh_double + 0.1));
        int left = (int) (Math.round(dw_double - 0.1));
        int right = (int) (Math.round(dw_double + 0.1));

        // ???????????????????????????size?????????
        Core.copyMakeBorder(out, out, top, bottom, left, right, Core.BORDER_CONSTANT, new Scalar(114, 114, 114));

    }

    public int clamp(int value, int min, int max) {
        // ???value?????????[min,max]?????????
        if (value > max) {
            value = max;

        } else if (value < min) {
            value = min;
        }

        return value;

    }

    private void draw(Integer class_id, Float confidence, int top_x, int top_y, int bottom_x, int bottom_y, Mat frame) {
        // ?????????????????????
        int thickness = Integer.max((int) ((frame.cols() + frame.rows()) / 2 * 0.003), 2);
        Imgproc.rectangle(frame, new Point(top_x, top_y), new Point(bottom_x, bottom_y),
                new Scalar(colorList[class_id]), thickness);

        // ???????????????????????????2?????????
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
            // ????????????????????????????????????????????????????????????
            double confidence = out.get(i, 4)[0];
            if (confidence > conf_threshold) {
                // ????????????????????????
                Mat scores = out.row(i).colRange(5, out.cols());
                Core.MinMaxLocResult result = Core.minMaxLoc(scores);
                int cls_id = (int) result.maxLoc.x;
                classIds.add(cls_id);

                // ?????????????????? * ???????????????????????????????????????,??????????????????NMSBoxes.
                confidences.add((float) (confidence * result.maxVal));


                //???????????????x,?????????y???????????????????????????cv2.dnn.NMSBoxes????????????????????????x????????????y?????????????????????

                // opencv???????????????????????????height????????????width; ?????????y????????????x;
                int x_center = (int) out.get(i, 0)[0];
                int y_center = (int) out.get(i, 1)[0];
                int width = (int) out.get(i, 2)[0];
                int height = (int) out.get(i, 3)[0];
                // + cls_id * max_wh ???????????????offset??????????????????????????????????????????
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

        // ???????????????????????????:???????????????box??????????????????
        Dnn.NMSBoxes(boxs, confis, (float) conf_threshold, (float) iou_threshold, idx_mat);

        // ????????????????????????????????????
        if (idx_mat.total() > 0) {
            int[] indices = idx_mat.toArray();
            float gain = Float.min((float) dst.rows() / (float) frame.rows(), (float) dst.cols() / (float) frame.cols());
            int x_pad = (int) (dst.cols() - frame.cols() * gain) / 2;
            int y_pad = (int) (dst.rows() - frame.rows() * gain) / 2;


            for (int i = 0; i < indices.length; ++i) {
                int box_idx = indices[i]; // box?????????
                Rect2d box = boxes.get(box_idx);
                int cls_id = classIds.get(box_idx); // ?????????

                // ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
                int top_x = (int) (((box.x - cls_id * max_wh) - x_pad) / gain);
                int top_y = (int) (((box.y - cls_id * max_wh) - y_pad) / gain);
                int bottom_x = (int) (top_x + box.width / gain);
                int bottom_y = (int) (top_y + box.height / gain);

                // ????????????????????????????????????????????????
                top_x = clamp(top_x, 0, frame.cols());
                bottom_x = clamp(bottom_x, 0, frame.cols());
                top_y = clamp(top_y, 0, frame.rows());
                bottom_y = clamp(bottom_y, 0, frame.rows());


                // ???????????????
                draw(classIds.get(box_idx), confidences.get(box_idx), top_x, top_y, bottom_x, bottom_y, frame);
            }
        }


    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        // ????????????
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

        // ???????????????, frame??????????????????CV_8UC3?????????8?????????????????????????????????3
        Mat frame = inputFrame.rgba();

        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
        Mat dst = new Mat();
        letterbox(frame, dst, IN_WIDTH, IN_HEIGHT);

        // ??????
        Mat blob = Dnn.blobFromImage(dst, 1 / 255.0);
        net.setInput(blob);
        Mat detections = net.forward(); //detections???????????????32FC1:32???float?????????1

        // ???????????????????????????
        detections = detections.reshape(0, (int) detections.size().width);
        post_process(frame, detections, dst, conf_thres, iou_thres, max_wh);

        return frame;
    }
}