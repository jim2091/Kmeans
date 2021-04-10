# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 15:36:14 2020

@author: Leejeongbin
"""


import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs



X, y = make_blobs(random_state=1)

K=5

m=len(X)
Centroids=np.array([]).reshape(2,0)

#랜덤한 centroid를 설정
for i in range(K):
    rand=rd.randint(0,m-1)
    Centroids=np.c_[Centroids,X[rand]] #np.c_[,]두 개의 1차원 배열을 칼럼으로 세로로 붙이기
    
    
    
#임의로 선택한 k개의 클러스터 중심과 개별 데이터 사이의 거리를 계산
n_iter=100

Output={} #출력값 초기화(딕셔너리)

for n in range(n_iter):
    # 거리 초기화
    Distance=np.array([]).reshape(m,0) #(100,0)크기의 배열 생성
    
    for k in range(K):
        temp_dist=np.sum((X-Centroids[:,k])**2,axis=1)
        #형상이 다른 두 배열의 계산이지만 Centroids의 차원 크기가 1이라 브로드캐스팅이 일어나 계산이 가능했다..!
        #자세한건 https://sacko.tistory.com/16 여기서 다시보자
        Distance=np.c_[Distance,temp_dist]
    #k번 반복되어 (100,k)의 배열이 만들어졌다. 각 centroids들과 각 X사이의 거리분산을 의미한다.
        
    C=np.argmin(Distance,axis=1)+1
    #argmin은 최소값의 색인. 따라서 1~3의 값을 갖는 (100,)의 배열이 생성. 각 X를 클러스터들에 할당한 정보.
    
    Y={} #출력값 y의 임시 딕셔너리 생성.
    for k in range(K):
        Y[k+1]=np.array([]).reshape(2,0)
    for i in range(m):
        Y[C[i]]=np.c_[Y[C[i]],X[i]]
    #Y는 (2,0)으로 위에서 만들어뒀고, X는 (100,2)로 만들어 져있었다. 이걸 아까 해봤듯이 Y에 계속 X의 i번째 열을 옆으로 붙이는 작업
    #결과적으로 클러스터별로 분류된 X마다의 Y의 정보.
    
    for k in range(K):
        Y[k+1]=Y[k+1].T
    #바로 앞에서 각 클러스터별로 (2,?)로 옆으로 쭈욱 있던 자료를 모양을 (?,2)로 바꿔주는 것.
    #진작에 (?,2)모양으로 만드는건 안되는거였을까.. 일단 해보고 시간이 된다면 시도해보자.
    
    for k in range(K):
        Centroids[:,k]=np.mean(Y[k+1],axis=0)
        
    Output=Y
    
plt.scatter(X[:,0],X[:,1],c='black',label='unclustered data')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.title('Plot of data points')
plt.show()

color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(K):
    plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(Centroids[0,:],Centroids[1,:],s=300,c='yellow',label='Centroids')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.show()
        
# 이 상태로 K=3으로 여러번 돌려본 결과 1:1:1, 0:100:0, 1:4:1로 클러스터링 되는 모습을 보였다.
# 대부분 1:1:1이었고 간혹 나머지 둘이 나오는 것으로 보아 나머지 둘이 지역최적해로 수렴한 상태인 것 같다.
# k를 5, 7 이런식으로 늘려보며 위의 과정을 반복하여 관찰해 보았는데 3일때에 비해 다양한 결과가
# 좀 더 일정하지 않게..?(3일때는 1:1:1인 경우가 압도적으로 많았음)나왔다.
# 일단 운이 좋게도 k=3으로 클러스터링 했을 경우 가장 예쁘게 잘 클러스터링이 된 모양이다.
# 지금 이렇게 직접 몇번 돌려보고 직접 분석한 과정을 또 코드로 잘 할 수 있을 거 같았지만 아직은 잘 모르겠다.
