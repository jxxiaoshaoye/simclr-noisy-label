# simclr-noisy-label
**simclr:**

pretain models an be find in dev branch.

For a quick start:

sh setup.sh || python3 -m pip install -r requirements.txt 

conda activate simclr

如果要训练新的模型：所有的模型都定义在module文件夹里的simclr.py

python main.py

想测试某个模型学完的representation：（transfer learning，frozen representation）

python testrepresentation.py

想测试某个模型学完的representation的表现：(supervised learning not frozen representation)

python phase2.py

**co-teaching:**

python -m testing.logistic_regression with dataset=CIFAR10 (CIFAR100)

(参数在config.yaml里面修改)
