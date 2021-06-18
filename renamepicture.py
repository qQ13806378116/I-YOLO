import glob
import os

inputPath = r'F:\yolo-eff\pyimgsaliency-master1\build\lib\pyimgsaliency\ronghe'
fileList = glob.glob(inputPath + '/*')

cnt = 5000
for path in fileList:
    os.rename(path, inputPath+'/'+ str(cnt)+'.jpg')
    cnt +=1