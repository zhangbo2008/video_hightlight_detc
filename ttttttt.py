import ffmpeg
(
    ffmpeg
    .input('somevideo/Bhxk-O1Y7Ho.mp4')
    .filter('fps', fps=0.5)
    .output('dummy2.mp4')
    .run()
)


print(1)