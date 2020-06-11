# simclr-noisy-label
##simclr:

git clone https://github.com/spijkervet/SimCLR.git && cd SimCLR

wget https://github.com/Spijkervet/SimCLR/releases/download/1.2/checkpoint_100.tar   #这个是resnet50的

sh setup.sh || python3 -m pip install -r requirements.txt 

conda activate simclr

如果要训练新的模型：所有的模型都定义在module文件夹里的simclr.py
改好main.py 和config.yaml里的参数之后，python main.py
想测试某个模型学完的representation是不是好：（transfer learning，frozen representation）

python testrepresentation.py

想测试某个模型学完的representation的表现：(supervised learning not frozen representation)

python phase2.py

##co-teaching:

python -m testing.logistic_regression with dataset=CIFAR10 model_path=. 

(参数在config.yaml里面修改)
