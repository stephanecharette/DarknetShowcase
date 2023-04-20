#include <MoveDetect.hpp>
#include <DarkHelp.hpp>
#include <filesystem>
#include <random>


/* This code was written for a 1-time presentation given to a group of high school students.  I needed an application
 * to quickly demo both objects detection and movement detection.  It is not great code!  It uses OpenCV's HighGUI
 * which is normally used as a quick-and-dirty debug tool.  I apologize in advance if this application doesn't look
 * right on your screen, or if it crashes, since a lot of the rectangle coordinates are hard-coded and were not tested
 * with different monitor resolutions and webcam dimensions.
 */


int main(int argc, char *argv[])
{
	std::cout << "Webcam Showcase" << std::endl;

	const cv::Scalar black(0, 0, 0);
	const cv::Scalar white(255, 255, 255);
	const cv::Scalar light_blue(222, 154, 0);

	std::default_random_engine random_engine;

	std::string input_filename		= "/dev/video0";
	std::string nn_cfg_filename		= "../yolov4-tiny.cfg";
	std::string nn_names_filename	= "../coco.names";
	std::string nn_weights_filename	= "../yolov4-tiny.weights";

	if (argc != 1 and
		argc != 2 and
		argc != 4 and
		argc != 5)
	{
		std::cout
			<< "Usage:"																		<< std::endl
			<< ""																			<< std::endl
			<< "\t" << argv[0]																<< std::endl
			<< "\t" << argv[0] << " <video>"												<< std::endl
			<< "\t" << argv[0] << " <cfg> <names> <weights>"								<< std::endl
			<< "\t" << argv[0] << " <cfg> <names> <weights> <video>"						<< std::endl
			<< ""																			<< std::endl
			<< "When not specified, the video will default to /dev/video0."					<< std::endl
			<< "When not specified, the neural network will default to MSCOCO YOLOv4-tiny."	<< std::endl
			<< "The <video> parameter can be a webcam device or a video file."				<< std::endl
			<< ""																			<< std::endl;

		throw std::invalid_argument("invalid number of arguments");
	}

	if (argc == 2)
	{
		// if we only have 1 parm, then that must be our input (webcam or video)
		input_filename = argv[1];
	}
	if (argc == 4 or argc == 5)
	{
		nn_cfg_filename		= argv[1];
		nn_names_filename	= argv[2];
		nn_weights_filename	= argv[3];
	}
	if (argc == 5)
	{
		// last parameter must be the webcam or video
		input_filename = argv[4];
	}

	DarkHelp::NN nn(nn_cfg_filename, nn_names_filename, nn_weights_filename);
	nn.config.sort_predictions				= DarkHelp::ESort::kAscending;
	nn.config.threshold						= 0.2;
	nn.config.include_all_names				= true;
	nn.config.names_include_percentage		= true;
	nn.config.annotation_auto_hide_labels	= false;
	nn.config.annotation_font_scale			= 1.0;
	nn.config.annotation_line_thickness		= 2;
	nn.config.annotation_include_duration	= false;
	nn.config.annotation_include_timestamp	= false;
	nn.config.annotation_pixelate_enabled	= false;
	nn.config.annotation_pixelate_size		= 25;
	nn.config.snapping_enabled				= false;
	nn.config.enable_tiles					= false;

	// see if we have any classes for "person" or "people"
	for (size_t idx = 0; idx < nn.names.size(); idx ++)
	{
		const auto & name = nn.names.at(idx);

		if (name == "person" or name == "people")
		{
			nn.config.annotation_pixelate_classes.insert(idx);
			nn.config.annotation_pixelate_enabled = true;
		}
	}

	// need to figure out where tools like xrandr and lshw get the screen dimensions...but for now
	// since this is a 1-time tool I suspect I'll only ever run once for the demo, hard-code the
	// display dimensions to 1920 x 1080 which is what this laptop uses
	const cv::Size screen_dimensions(1920, 1080);

	/* The main windows are:
	 *
	 *		1) Object detection from DarkHelp/Darknet/YOLO
	 *		2) Movement detection from MoveDetect
	 *
	 * Smaller windows are:
	 *
	 *		3) Detection mask from MoveDetect
	 *
	 * Let's say the main images are 800x600.  So if the desktop is 1920x1080, that means horizontally
	 * we still have 320 horizontally and 480 vertically for smaller windows.
	 */
	const cv::Size large_image_dimensions(900, 506); // 1.7777777778 aspect ratio
	const cv::Size small_image_dimensions(360, 202);

	const int horizontal_gaps			= std::round((screen_dimensions.width	- large_image_dimensions.width * 2.0f							) / 3.0f);
	const int vertical_gaps				= std::round((screen_dimensions.height	- large_image_dimensions.height - small_image_dimensions.height	) / 3.0f);
	const int number_of_small_images	= screen_dimensions.width / small_image_dimensions.width;
	const int small_image_gap			= std::round((screen_dimensions.width - number_of_small_images * small_image_dimensions.width) / (number_of_small_images + 1.0));
	const float small_image_aspect_ratio = static_cast<float>(small_image_dimensions.width) / static_cast<float>(small_image_dimensions.height);

	const cv::Rect object_detection_rect	(cv::Point(horizontal_gaps, vertical_gaps), large_image_dimensions);
	const cv::Rect movement_detection_rect	(cv::Point(large_image_dimensions.width + horizontal_gaps * 2, vertical_gaps), large_image_dimensions);

	std::vector<cv::Rect> small_window_rects;
	for (int i = 0; i < number_of_small_images; i ++)
	{
		cv::Rect r(
			small_image_gap * (i + 1) + small_image_dimensions.width * i,
			large_image_dimensions.height + vertical_gaps * 2,
			small_image_dimensions.width,
			small_image_dimensions.height);
		small_window_rects.push_back(r);
	}
	const cv::Rect movement_mask_rect = small_window_rects.at(number_of_small_images - 1);
	const cv::Rect fps_rect(cv::Point(10, small_window_rects[0].y + small_window_rects[0].height + 20), cv::Size(75, 15));
	const cv::Point fps_point(fps_rect.x, fps_rect.y + fps_rect.height);

	cv::Mat output(screen_dimensions, CV_8UC3, light_blue);
	cv::putText(output, "object detection with Darknet/YOLO", cv::Point(object_detection_rect.x		, object_detection_rect		.y - 10), cv::FONT_HERSHEY_PLAIN, 2.0, black, 1, cv::LINE_AA);
	cv::putText(output, "PSNR (Peak Signal to Noise Ratio)"	, cv::Point(movement_detection_rect.x	, movement_detection_rect	.y - 10), cv::FONT_HERSHEY_PLAIN, 2.0, black, 1, cv::LINE_AA);
	cv::putText(output, "movement detection mask"			, cv::Point(movement_mask_rect.x		, movement_mask_rect		.y - 10), cv::FONT_HERSHEY_PLAIN, 1.5, black, 1, cv::LINE_AA);

	cv::VideoCapture cap(input_filename);
	if (cap.isOpened() == false)
	{
		throw std::runtime_error("failed to open " + input_filename);
	}

	if (input_filename.find("/dev/video") == 0)
	{
		cap.set(cv::VideoCaptureProperties::CAP_PROP_FPS			, 30.0	);
		cap.set(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH	, large_image_dimensions.width);
		cap.set(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT	, large_image_dimensions.height);
//		cap.set(cv::VideoCaptureProperties::CAP_PROP_BUFFERSIZE		, 10.0	);
	}

	const double input_fps			= cap.get(cv::VideoCaptureProperties::CAP_PROP_FPS			);
	const double original_width		= cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH	);
	const double original_height	= cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT	);
	const size_t frame_length_ns	= std::round(1000000000.0 / input_fps);

#if 0
	cv::VideoWriter video_writer;
	video_writer.open("output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), input_fps, screen_dimensions);
	if (not video_writer.isOpened())
	{
		throw std::runtime_error("failed to open output video file");
	}
#endif

	std::cout
		<< ""																									<< std::endl
		<< "Input filename ....... " << input_filename															<< std::endl
		<< "Input dimensions ..... " << original_width					<< "x" << original_height				<< std::endl
		<< "Screen dimensions .... " << screen_dimensions.width			<< "x" << screen_dimensions.height		<< std::endl
		<< "Large image size ..... " << large_image_dimensions.width	<< "x" << large_image_dimensions.height	<< std::endl
		<< "Small image size ..... " << small_image_dimensions.width	<< "x" << small_image_dimensions.height	<< std::endl
		<< "Small images ......... " << number_of_small_images													<< std::endl
		<< "Horizontal gaps ...... " << horizontal_gaps					<< " pixels"							<< std::endl
		<< "Vertical gaps ........ " << vertical_gaps					<< " pixels"							<< std::endl
		<< "Small image gaps ..... " << small_image_gap					<< " pixels"							<< std::endl
		<< "Frame rate ........... " << input_fps						<< " FPS"								<< std::endl
		<< "Frame interval ....... " << frame_length_ns					<< " nanoseconds"						<< std::endl
		<< "Frame interval ....... " << (frame_length_ns / 1000000.0)	<< " milliseconds"						<< std::endl;

	MoveDetect::Handler handler;
	handler.contours_enabled			= true;
	handler.contours_size				= 10;
	handler.mask_enabled				= true;
	handler.bbox_enabled				= false;
	handler.bbox_size					= 10;
	handler.line_type					= cv::LINE_AA;
	handler.key_frame_frequency			= input_fps / 3;
	handler.number_of_control_frames	= 2;

	const std::chrono::high_resolution_clock::duration		frame_duration			= std::chrono::nanoseconds(frame_length_ns);
	const std::chrono::high_resolution_clock::time_point	start_time				= std::chrono::high_resolution_clock::now();
	auto next_frame_time_point														= start_time + frame_duration;
	auto now																		= start_time;

	std::time_t previous_fps_tt = 0;
	size_t previous_fps_frame_index = 0;

	int idx_of_small_image_to_update = number_of_small_images - 1;
	float small_image_confidence = 0.0f;

	size_t frame_index = 0;
	while (cap.isOpened())
	{
		cv::Mat frame;
		cap >> frame;
		if (frame.empty())
		{
			break;
		}
		frame_index ++;

		handler.detect(frame);

		cv::Mat mask;
		cv::resize(handler.mask, mask, small_image_dimensions, 0.0, 0.0, cv::INTER_LINEAR);
		cv::cvtColor(mask, output(movement_mask_rect), cv::COLOR_GRAY2BGR);

		cv::resize(handler.output, output(movement_detection_rect), large_image_dimensions, 0.0, 0.0, cv::INTER_LINEAR);

		nn.predict(frame);

		cv::resize(nn.annotate(), output(object_detection_rect), large_image_dimensions, 0.0, 0.0, cv::INTER_LINEAR);

		// every 5 seconds we update the FPS
		const auto tt = std::time(nullptr);
		if (tt >= previous_fps_tt + 5)
		{
			const float time_elapsed_in_seconds = tt - previous_fps_tt;
			const float frames_elapsed = frame_index - previous_fps_frame_index;

			// only do the calculations if the numbers make sense
			if (time_elapsed_in_seconds > 0 and
				time_elapsed_in_seconds < 10 and
				frames_elapsed > 0 and
				frames_elapsed < 500)
			{
				std::stringstream ss;
				ss << std::fixed << std::setprecision(1) << (frames_elapsed / time_elapsed_in_seconds) << " FPS ";
				const auto fps = ss.str();
				std::cout << "\r" << fps << std::flush;
				output(fps_rect) = light_blue;
				cv::putText(output, fps, fps_point, cv::FONT_HERSHEY_PLAIN, 1.0, white, 1, cv::LINE_AA);
			}
			previous_fps_tt = tt;
			previous_fps_frame_index = frame_index;

			// make one of the thumbnails available
			idx_of_small_image_to_update ++;
			if (idx_of_small_image_to_update >= (number_of_small_images - 1))
			{
				idx_of_small_image_to_update = 0;
			}
			small_image_confidence = 0.0f;
		}

		if (not nn.prediction_results.empty())
		{
			std::uniform_int_distribution<int> distribution(0, nn.prediction_results.size() - 1);
			const auto idx = distribution(random_engine);

			const auto & prediction = nn.prediction_results.at(idx);

			if (prediction.best_probability > small_image_confidence + 0.2)
			{
				small_image_confidence = prediction.best_probability;

				// Modify the x, y, w, and h until the aspect ratio matches the small window.
				// For example, 360x270 gives us an aspect ratio of 1.333333.
				//
				// But if our prediction is 360x100, then our aspect ratio is 3.6, meaning we need to grow the height.

				auto r = prediction.rect;
				r.x -= 25;
				r.y -= 25;
				r.width += 50;
				r.height += 50;

				const float aspect_ratio = static_cast<float>(r.width) / static_cast<float>(r.height);
				if (aspect_ratio < small_image_aspect_ratio)
				{
					// need to fix the width
					const int desired_width = std::round(r.height / small_image_aspect_ratio);
					r.x -= desired_width / 2;
					r.width += desired_width;
				}
				else if (aspect_ratio > small_image_aspect_ratio)
				{
					// need to fix the height
					const int desired_height = std::round(r.width / small_image_aspect_ratio);
					r.y -= desired_height / 2;
					r.height += desired_height;
				}
				if (r.x < 0)						r.x = 0;
				if (r.x + r.width > frame.cols)		r.width = frame.cols - r.x;
				if (r.y < 0)						r.y = 0;
				if (r.y + r.height > frame.rows)	r.height = frame.rows - r.y;

				const auto colour_idx = prediction.best_class % nn.config.annotation_colours.size();
				cv::rectangle(frame, prediction.rect, nn.config.annotation_colours.at(colour_idx), 2, cv::LINE_AA);

				cv::Mat mat;
				cv::resize(frame(r), mat, small_image_dimensions, 0.0, 0.0, cv::INTER_LINEAR);
				mat.copyTo(output(small_window_rects[idx_of_small_image_to_update]));
			}
		}

		cv::imshow("output", output);

#if 0
		video_writer.write(output);
#endif

		// wait for the right amount of time so we have the correct FPS
		now = std::chrono::high_resolution_clock::now();
		const int milliseconds_to_wait = std::chrono::duration_cast<std::chrono::milliseconds>(next_frame_time_point - now).count();
		const auto key = cv::waitKey(std::max(1, milliseconds_to_wait));
		if (key == 27)
		{
			break;
		}
		next_frame_time_point += frame_duration;
		if (now > next_frame_time_point)
		{
			// we've fallen too far behind, reset the time we need to show the next frame
			next_frame_time_point = now + frame_duration;
		}
	}

	// ESC was pushed, or the camera is no longer available
	std::cout << std::endl;

	const std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

	std::cout
		<< std::endl
		<< "-> processed " << frame_index << " frames" << std::endl
		<< "-> time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " milliseconds" << std::endl;

	return 0;
}
