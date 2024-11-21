from train import main_process
from utils import conf_list
from torch.cuda import empty_cache
for i in conf_list:
    try:
        print(i.getSaveName('')[:-1])
        main_process(i)
    except Exception as e:
        empty_cache()
        with open('log.txt','a') as f:
            f.write(i.getSaveName('')+','+str(e)+'\n')
        continue