ROOT = "/home/malbranke/cas9_definitive"
DATA = f"{ROOT}/data"

gammas = [1]+ [1.4**i/1000 for i in range(50)]+[0]+[2*1.05**i for i in range(50)]+[1e-7*1.05**i for i in range(100)]+[1.4**i/1000 for i in range(50)]+[0.01*1.05**i for i in range(150)]+[0.0001*1.05**i for i in range(100)]+[1e-5*1.05**i for i in range(50)]+[20*1.08**i for i in range(50)]
