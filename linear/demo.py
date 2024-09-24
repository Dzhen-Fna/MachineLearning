def judge_1(x1,x2,x3):
    judge = 3*x1 - x2 + 2*x3
    if  judge<=7:
        return True
    else:
        print('1',judge)
        return False
    
def judge_2(x1,x2):
    judge = -2*x1 + 4*x2
    if judge<= 12:
        return True
    else:
        print('2',judge)
        return False
    
def judge_3(x1,x2,x3):
    judge =  -4*x1 + 3*x2 + 8*x3
    if judge<= 10:
        return True
    else:
        print('3',judge)
        return False
def culc(x1,x2,x3):
    if judge_1(x1,x2,x3) and judge_2(x1,x2) and judge_3(x1,x2,x3):
        print(-1*x1 + 3*x2 + x3)        

culc(78/25,114/25,11/10)