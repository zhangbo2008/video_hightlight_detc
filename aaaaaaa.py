import ffmpeg  # 安装 https://oldtang.com/976.html
 
info = ffmpeg.probe('somevideo/Bhxk-O1Y7Ho.mp4')  # 获取视频信息，其他接口可参考具体文档
print(1)