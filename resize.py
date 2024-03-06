from moviepy.editor import VideoFileClip
from screeninfo import get_monitors

def resize_video_to_screen_resolution(video_path, output_path):
    # Get the screen resolution from the first monitor
    monitor = get_monitors()[0]
    screen_width, screen_height = monitor.width, monitor.height

    # Load the original video
    clip = VideoFileClip(video_path)

    # Calculate the new size to maintain the aspect ratio
    aspect_ratio = clip.size[0] / clip.size[1]
    if screen_width / screen_height > aspect_ratio:
        # Screen is wider than the video aspect ratio
        new_height = screen_height
        new_width = int(aspect_ratio * screen_height)
    else:
        # Screen is narrower than the video aspect ratio
        new_width = screen_width
        new_height = int(screen_width / aspect_ratio)
    
    # Resize the video
    resized_clip = clip.resize(newsize=(new_width, new_height))
    
    # Write the resized video to the output file
    resized_clip.write_videofile(output_path, audio_codec='aac')

# Example usage
if __name__ == "__main__":
    video_path = "./final.mp4"
    output_path = "./final_resized.mp4"
    resize_video_to_screen_resolution(video_path, output_path)
