import struct 
import cmath as cm
import matplotlib.pyplot as plt 
import math
import glob
import os
import re

#バイナリファイル読み込み用関数
def read_binaryshort(filename):
    f=open(filename, 'rb')
    data=[]
    while True:
        #2byteごと読み込む
        temp = f.read(2)
        if not temp:
            break
        data.append(struct.unpack('h',temp)[0])
    return data

#256サンプリング点ごとに離散フーリエ変換を行い対数パワースペクトルのリストを返す
def dft(data):
    res = []
    N = 256
    i=0
    while True:
        if (i+1)*N > len(data):
            break
        temp = data[i*N:(i+1)*N]
        x = []
        for k in range(N):
            w = cm.exp(-1j * 2 * cm.pi * k / N)
            X_k = 0
            for n in range(N):
                X_k += temp[n] * (w ** n)
            x.append(math.log(X_k.real**2+X_k.imag**2))
        res.append(x[1:int(N/2)])
        i+=1
    return res

#dftされたデータから多次元正規分布を求める
def find_normal_distribution(data):
    u,s = [],[]
    for i in range(len(data[0])):
        #平均を求める
        sum=0
        for j in range(len(data)):
            sum += data[j][i]
        average  = sum / len(data)
        #分散を求める
        sum = 0
        for j in range(len(data)):
            sum += (data[j][i] - average)**2
        variance = sum/len(data)

        u.append(average)
        s.append(variance)
    return u, s

#確率計算用の関数
def calc_probability(u,s,data):
    const = len(s)*math.log(2*math.pi)
    for i in s:
        const += math.log(i)

    prob = 0
    for l in range(len(data)):
        temp = const
        for i in range(len(u)):
            xmm = data[l][i]-u[i]
            temp += xmm*xmm/s[i]
        prob += -0.5*temp
    
    return prob
        
if __name__ == "__main__":

    u,s ,label= [],[],[]
    
    for f in glob.glob('data/train*.raw'):
        spec = dft(read_binaryshort(f))
        u_temp,s_temp = find_normal_distribution(spec)
        u.append(u_temp)
        s.append(s_temp)

        filename = os.path.split(f)[1]
        print('loaded '+ filename)
        #train_*.rawの*の部分を抽出しておく
        label.append(re.search('(?<=train_).*(?=\.raw)', filename).group())

    #学習した平均値ベクトルのプロット
    for i,l in zip(u,label):
        plt.plot(i,label = l)
    plt.xlabel('Frequency index of k')
    plt.ylabel('Log-power')
    plt.legend()
    plt.show()

    for f in glob.glob('data/test*.raw'):
        max =  -100000000000
        ind = 0
        data = dft(read_binaryshort(f))
        #学習した各モデルについて確率を求める
        print(f+'__________')
        for i in range(len(u)):
            prob = calc_probability(u[i],s[i],data)
            print(label[i],prob)
            if max < prob:
                max = prob
                ind = i
        print('result : '+label[ind])

    
    

