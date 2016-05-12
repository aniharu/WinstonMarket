__author__ = 'umeco'
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from scipy import linalg as LA
import sys

class DataCreate:
    def __init__(self,highver,midver,lowver,flag):
        self.time=time.time()
        self.highver=highver
        self.midver=midver
        self.lowver=lowver
        if flag==1:
            self.data=np.genfromtxt("../test_2.csv",skip_header=1,delimiter=',',dtype='float32')
        else:
            self.data=np.genfromtxt("../mycsv/mytrain"+str(self.highver)+"."+str(self.midver)+"."+str(self.lowver)+".csv",delimiter=',',skip_header=1,dtype='float32')
        self.header="Id,Feature_1,Feature_2,Feature_3,Feature_4,Feature_5,Feature_6,Feature_7,Feature_8,Feature_9,Feature_10,Feature_11,Feature_12,Feature_13,Feature_14,Feature_15,Feature_16,Feature_17,Feature_18,Feature_19,Feature_20,Feature_21,Feature_22,Feature_23,Feature_24,Feature_25,Ret_MinusTwo,Ret_MinusOne,Ret_2,Ret_3,Ret_4,Ret_5,Ret_6,Ret_7,Ret_8,Ret_9,Ret_10,Ret_11,Ret_12,Ret_13,Ret_14,Ret_15,Ret_16,Ret_17,Ret_18,Ret_19,Ret_20,Ret_21,Ret_22,Ret_23,Ret_24,Ret_25,Ret_26,Ret_27,Ret_28,Ret_29,Ret_30,Ret_31,Ret_32,Ret_33,Ret_34,Ret_35,Ret_36,Ret_37,Ret_38,Ret_39,Ret_40,Ret_41,Ret_42,Ret_43,Ret_44,Ret_45,Ret_46,Ret_47,Ret_48,Ret_49,Ret_50,Ret_51,Ret_52,Ret_53,Ret_54,Ret_55,Ret_56,Ret_57,Ret_58,Ret_59,Ret_60,Ret_61,Ret_62,Ret_63,Ret_64,Ret_65,Ret_66,Ret_67,Ret_68,Ret_69,Ret_70,Ret_71,Ret_72,Ret_73,Ret_74,Ret_75,Ret_76,Ret_77,Ret_78,Ret_79,Ret_80,Ret_81,Ret_82,Ret_83,Ret_84,Ret_85,Ret_86,Ret_87,Ret_88,Ret_89,Ret_90,Ret_91,Ret_92,Ret_93,Ret_94,Ret_95,Ret_96,Ret_97,Ret_98,Ret_99,Ret_100,Ret_101,Ret_102,Ret_103,Ret_104,Ret_105,Ret_106,Ret_107,Ret_108,Ret_109,Ret_110,Ret_111,Ret_112,Ret_113,Ret_114,Ret_115,Ret_116,Ret_117,Ret_118,Ret_119,Ret_120,Ret_121,Ret_122,Ret_123,Ret_124,Ret_125,Ret_126,Ret_127,Ret_128,Ret_129,Ret_130,Ret_131,Ret_132,Ret_133,Ret_134,Ret_135,Ret_136,Ret_137,Ret_138,Ret_139,Ret_140,Ret_141,Ret_142,Ret_143,Ret_144,Ret_145,Ret_146,Ret_147,Ret_148,Ret_149,Ret_150,Ret_151,Ret_152,Ret_153,Ret_154,Ret_155,Ret_156,Ret_157,Ret_158,Ret_159,Ret_160,Ret_161,Ret_162,Ret_163,Ret_164,Ret_165,Ret_166,Ret_167,Ret_168,Ret_169,Ret_170,Ret_171,Ret_172,Ret_173,Ret_174,Ret_175,Ret_176,Ret_177,Ret_178,Ret_179,Ret_180,Ret_PlusOne,Ret_PlusTwo,Weight_Intraday,Weight_Daily\n"
    def print_datanum(self,feature):#featureのデータ集計
        dict={"nan":0}
        for i in range(len(self.data)):
            if self.data[i,feature] == "nan":
                dict["nan"]=dict["nan"]+1
            elif self.data[i,feature] not in dict:
                dict[self.data[i,feature]]=1
            else:
                dict[self.data[i,feature]]=dict[self.data[i,feature]]+1
        print(dict)
    def fill_average(self,feature):#featureの平均値による欠損値埋め
        sum=0
        num=0
        for i in range(len(self.data)):
            if np.isnan(self.data[i,feature])==False:
                sum+=self.data[i,feature]
                num+=1
        sum/=num
        for i in range(len(self.data)):
            if np.isnan(self.data[i,feature]):
                self.data[i,feature]=sum
    def fill_random(self,feature):#featureの離散値を集め欠損値に対してランダムで割り振る
        memo=[]
        for i in range(len(self.data)):
            if np.isnan(self.data[i,feature])==False:
                memo.append(self.data[i,feature])
        for i in range(len(self.data)):
            if np.isnan(self.data[i,feature]):
                self.data[i,feature]=memo[random.ranint(0,len(memo))]
    def print_Avermse(self,feature):#featureの平均値を算出し、そのRMSEを表示する
        sum=0
        num=0
        rmse=0
        wmse=0
        for i in range(len(self.data)):
            if np.isnan(self.data[i,feature])==False:
                sum+=self.data[i,feature]
                num+=1
        sum/=num
        for i in range(len(self.data)):
            if np.isnan(self.data[i,feature])==False:
                rmse+=pow(sum-self.data[i,feature],2)/num
        rmse=pow(rmse,0.5)
        for i in range(len(self.data)):
            if feature == 207 or feature ==208:
                wmse+=self.data[i,210]*abs(self.data[i,feature])
            else:
                wmse+=self.data[i,209]*abs(self.data[i,feature])
        wmse/=len(self.data)
        print("平均値:"+str(sum))
        print("RMSE:"+str(rmse))
        print("WMSE:"+str(wmse))
    def write_data(self):#リストからtrain.csvを書き出す
        f=open("../mycsv/mytrain"+str(self.highver)+"."+str(self.midver)+"."+str(self.lowver+1)+".csv","w")
        f.write(self.header)
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                if np.isnan(self.data[i,j])==False:
                    f.write(str(self.data[i,j]))
                if j!=len(self.data[0])-1:
                    f.write(",")
            f.write("\n")
        f.close()
    def write_test(self,N):#リストからN行Dataを書き出す
        f=open("../mycsv/test.csv","w")
        f.write(self.header)
        for i in range(N):
            for j in range(len(self.data[0])):
                if np.isnan(self.data[i,j])==False:
                    f.write(str(self.data[i,j]))
                if j!=len(self.data[0])-1:
                    f.write(",")
            f.write("\n")
        f.close()
    def create_specialdata(self):
        sum=0
        f=open("../mycsv/special.csv","w")
        f.write("Feature_1,Feature_2,Feature_3,Feature_4,Feature_5,Feature_6,Feature_7,Feature_8,Feature_9,Feature_10,Feature_11,Feature_12,Feature_13,Feature_14,Feature_15,Feature_16,Feature_17,Feature_18,Feature_19,Feature_20,Feature_21,Feature_22,Feature_23,Feature_24,Feature_25,Day-2,Day-1,Today,Day+1,Day+2\n")
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                if (j>=1 and j <=25)or j==26 or j==27 or j==207 or j==208:
                    f.write(str(self.data[i,j]))
                    if j!=208:
                        f.write(",")
                if j>=28 and j<=146:
                    sum+=self.data[i,j]
                    if j==146:
                        f.write(str(sum)+",")
                        sum=0
            f.write("\n")
        f.close()
    def enR(self,feature,elements,flag):
        x=self.data[:,elements[0]]
        for i in elements[1:]:
            x=np.c_[x,self.data[:,i]]
        y=self.data[:,feature]

        prex=x
        x=x[~np.isnan(y)]  #欠損値の存在する行を削除
        y=y[~np.isnan(y)]

        param={'alpha':[i/10 for i in range(1,10)],
               'l1_ratio':[i/10 for i in range(1,10)]}
        GS=GridSearchCV(ElasticNet(),param,cv=10,n_jobs=-1)
        GS.fit(x,y)
        print(GS.best_estimator_)


        en = ElasticNet(alpha=GS.best_estimator_.alpha,l1_ratio=GS.best_estimator_.l1_ratio)
        en.fit(x,y)
        print(en.coef_)

        score=cross_validation.cross_val_score(en,x,y,cv=10,scoring='mean_squared_error')
        print("ElasticNet:"+str(-1*score.mean())) #mean()では正規の値に-1が掛かったものが出てくるため

        if flag==1:
            for i in range(len(self.data)):
                if np.isnan(self.data[i,feature]):
                    self.data[i,feature]=en.predict(prex[i])
    def ridgeR(self,feature,elements,flag):
        x=self.data[:,elements[0]]
        for i in elements[1:]:
            x=np.c_[x,self.data[:,i]]
        y=self.data[:,feature]

        x=x[~np.isnan(y)]  #欠損値の存在する行を削除
        y=y[~np.isnan(y)]

        param={'alpha':[i/10 for i in range(1,10)]}
        GS=GridSearchCV(Ridge(),param,cv=10,n_jobs=-1)
        GS.fit(x,y)
        print(GS.best_estimator_)


        en = Ridge(alpha=GS.best_estimator_.alpha)
        en.fit(x,y)
        print(en.coef_)

        score=cross_validation.cross_val_score(en,x,y,cv=10,scoring='mean_squared_error')
        print("Ridge:"+str(-1*score.mean())) #mean()では正規の値に-1が掛かったものが出てくるため
    def lassoR(self,feature,elements):
        x=self.data[:,elements[0]]
        for i in elements[1:]:
            x=np.c_[x,self.data[:,i]]
        y=self.data[:,feature]

        x=x[~np.isnan(y)]  #欠損値の存在する行を削除
        y=y[~np.isnan(y)]

        param={'alpha':[i/10000 for i in range(1,10)]}
        GS=GridSearchCV(Lasso(),param,cv=10,n_jobs=-1)
        GS.fit(x,y)
        print(GS.best_estimator_)


        en = Lasso(alpha=GS.best_estimator_.alpha)
        en.fit(x,y)
        print(en.coef_)

        score=cross_validation.cross_val_score(en,x,y,cv=10,scoring='mean_squared_error')
        print("Lasso:"+str(-1*score.mean())) #mean()では正規の値に-1が掛かったものが出てくるため
    def printTime(self):
        print("\n実行時間: "+str(time.time()-self.time)+" sec")
    def fill_return(self): #returnの欠損値を埋める関数
        for i in range(len(self.data)):
            for j in range(28,147):
                if np.isnan(self.data[i,j]):
                    if j==28:
                        if np.isnan(self.data[i,j+1]):
                            self.data[i,j]=0
                        else:
                            self.data[i,j]=self.data[i,j+1]
                    else:
                        if np.isnan(self.data[i,j+1]):
                            self.data[i,j]=self.data[i,j-1]
                        else:
                            self.data[i,j]=(self.data[i,j-1]+self.data[i,j+1])/2
    def print_Zpredict(self):
        rmse=0
        wmse=0
        for i in range(len(self.data)):
            for j in range(147,207):
                rmse+=pow(self.data[i,j],2)
                wmse+=self.data[i,209]*abs(self.data[i,j])
            for j in [207,208]:
                rmse+=pow(self.data[i,j],2)
                wmse+=self.data[i,210]*abs(self.data[i,j])
        rmse/=len(self.data)*62
        rmse=pow(rmse,0.5)
        wmse/=len(self.data)*62
        print("RMSE: "+str(rmse))
        print("WMSE: "+str(wmse))
    def return_enR(self,elements):
        predict=np.ones((len(self.data),62))
        x=self.data[:,elements[0]]
        for i in elements[1:]:
            x=np.c_[x,self.data[:,i]]
        for target in range(147,209):
            print("目的変数:"+str(target-146))
            y=self.data[:,target]
            en = ElasticNet(alpha=0.5,l1_ratio=0.1)
            en.fit(x,y)
            for i in range(len(self.data)):
                predict[i,target-147]=en.predict(x[i])
        rmse=0
        wmse=0
        for i in range(len(self.data)):
            for j in range(147,207):
                rmse+=pow(predict[i,j-147]-self.data[i,j],2)
                wmse+=self.data[i,209]*abs(predict[i,j-147]-self.data[i,j])
            for j in [207,208]:
                rmse+=pow(predict[i,j-147]-self.data[i,j],2)
                wmse+=self.data[i,210]*abs(predict[i,j-147]-self.data[i,j])
        rmse/=len(self.data)*62
        rmse=pow(rmse,0.5)
        wmse/=len(self.data)*62
        print("RMSE: "+str(rmse))
        print("WMSE: "+str(wmse))
    def returnRe_enR(self,elements):#不安定な要素を抜いたver
        predict=np.ones((len(self.data),62))
        smoozing=np.zeros((len(self.data),24))
        graf_y=np.zeros(62)
        zero_y=np.zeros(62)
        rmse_y=np.zeros(62)
        graf_x=range(62)
        zero=0
        x=self.data[:,elements[2]]
        for num in range(len(self.data)):
            for i in range(24):
                if i==23:
                    for j in range(4):
                        smoozing[num,i]+=(self.data[num,i*5+j+28])
                    smoozing[num,i]/=4
                    continue
                for j in range(5):
                    smoozing[num,i]+=(self.data[num,i*5+j+28])
                smoozing[num,i]/=5
        for i in elements[3:]:
            if i==4 or i==10 or i==20 or (i>=28 and i<=146):
                continue
            x=np.c_[x,self.data[:,i]]
        for i in range(24):
            x=np.c_[x,smoozing[:,i]]
        for target in range(147,209):
            print("目的変数:"+str(target-146))
            y=self.data[:,target]
            en = ElasticNet(alpha=0.2,l1_ratio=0)
            en.fit(x,y)
            for i in range(len(self.data)):
                predict[i,target-147]=en.predict(x[i])
        rmse=0
        wmse=0
        for i in range(len(self.data)):
            for j in range(147,207):
                rmse+=pow(predict[i,j-147]-self.data[i,j],2)
                wmse+=self.data[i,209]*abs(predict[i,j-147]-self.data[i,j])
                graf_y[j-147]+=self.data[i,209]*abs(predict[i,j-147]-self.data[i,j])
                zero+=self.data[i,209]*abs(self.data[i,j])
                zero_y[j-147]+=self.data[i,209]*abs(self.data[i,j])
                rmse_y[j-147]+=pow(predict[i,j-147]-self.data[i,j],2)

            for j in [207,208]:
                rmse+=pow(predict[i,j-147]-self.data[i,j],2)
                wmse+=self.data[i,210]*abs(predict[i,j-147]-self.data[i,j])
                graf_y[j-147]+=self.data[i,210]*abs(predict[i,j-147]-self.data[i,j])
                rmse_y[j-147]+=pow(predict[i,j-147]-self.data[i,j],2)

        rmse/=len(self.data)*62
        rmse=pow(rmse,0.5)
        rmse_y/=len(self.data)
        for i in range(62):
            rmse_y[i]=pow(rmse_y[i],0.5)
        wmse/=len(self.data)*62
        zero/=len(self.data)*62
        graf_y/=len(self.data)
        zero_y/=len(self.data)
        print("RMSE: "+str(rmse))
        print("WMSE: "+str(wmse))
        print("ZERO WMSE: "+str(zero))
        print(en.coef_)
        plt.subplot(3,1,1)
        plt.plot(graf_x[:60],graf_y[:60],'b')
        plt.plot(graf_x[:60],zero_y[:60],'r')
        plt.legend(["回帰","０予測"],loc='lower right')
        plt.title("WMSE")
        plt.subplot(3,1,2)
        plt.title("RMSE")
        plt.plot(graf_x,rmse_y)
        plt.subplot(3,1,3)
        plt.title("RMSE(60)")
        plt.plot(graf_x[:60],graf_y[:60])
        plt.show()
    def nuralnet(self):
        data=np.genfromtxt("../mycsv/special.csv",dtype="float32",delimiter=",",skip_header=1)
        with tf.name_scope("test") as scope:
            x = tf.placeholder("float", shape=[None, 121])
            y_ = tf.placeholder("float", shape=[None, 2])
            W1 = tf.Variable(tf.truncated_normal([121,10],stddev=0.1)) # 正規分布でウェイトを初期化
            W2 = tf.Variable(tf.truncated_normal([10,2],stddev=0.1))
            sig_w = tf.Variable(tf.truncated_normal([121,2],stddev=0.1))
            #sig_w2 = tf.Variable(tf.truncated_normal([5,1],stddev=0.1))
            W3 = tf.Variable(tf.truncated_normal([2,2],stddev=0.01))
            b1 = tf.Variable(tf.zeros([10]))
            b2 = tf.Variable(tf.zeros([2]))
            sig_b = tf.Variable(tf.zeros([2]))
            y = tf.matmul(x, W1) + b1
            y2 = tf.matmul(y, W2) + b2
            sig_y = tf.nn.sigmoid(tf.matmul(x,sig_w)+sig_b)
            y3 = tf.matmul(tf.matmul(y2, sig_y),W3)

            loss = tf.reduce_mean(tf.square(y3 - y_))
            #optimizer = tf.train.GradientDescentOptimizer(0.2)
            optimizer = tf.train.FtrlOptimizer(0.5)
            #optimizer = tf.train.AdagradOptimizer(0.01)
            #optimizer = tf.train.AdamOptimizer()
            train = optimizer.minimize(loss)
            init = tf.initialize_all_variables()
            sess = tf.Session()
            sess.run(init)

            #Graph Data
            tf.scalar_summary('LOSS',loss)
            #tf.histogram_summary('Y',y)
            summary_op=tf.merge_all_summaries()
            summary_writer=tf.train.SummaryWriter('./',graph_def=sess.graph_def)

            print("反復開始")
            graphx=[]
            graphy=[]
            for step in range(10001):
                tmp = random.sample(range(len(data)), 100)
                tmp = self.data[tmp, :]
                x_tmp = tmp[:, 26:147]
                y_tmp = tmp[:, 207:209]
                sess.run(train, feed_dict={x: x_tmp, y_: y_tmp})
                if step%1000==0:
                    print(step,sess.run(W1),sess.run(b1))
                    tf.scalar_summary('LOSS',loss)
                    graphx.append(step)
                    graphy.append(self.RMSET(sess.run(y3, feed_dict={x: self.data[:, 26:147]})))
                summary_str=sess.run(summary_op,feed_dict={x: x_tmp, y_: y_tmp})
                summary_writer.add_summary(summary_str,step)

            predict = sess.run(y3, feed_dict={x: self.data[:, 26:147]})
            rmse = [0., 0.]
            wmse = [0., 0.]
            zmse = [0., 0.]

            for i in range(len(predict)):
                for j in range(2):
                    rmse[j] += pow(self.data[i, j + 207] - predict[i, j], 2)
                    zmse[j] += self.data[i, 210] * abs(self.data[i, j + 207])
                    wmse[j] += self.data[i, 210] * abs(self.data[i, j + 207] - predict[i, j])
            rmse[0] = pow(rmse[0] / len(predict), 0.5)
            rmse[1] = pow(rmse[1] / len(predict), 0.5)
            wmse[0] = wmse[0] / len(predict)
            wmse[1] = wmse[1] / len(predict)
            zmse[0] = zmse[0] / len(predict)
            zmse[1] = zmse[1] / len(predict)
            print("NuralRMSE Day1:%f\n" % rmse[0])
            print("NuralRMSE Day2:%f\n" % rmse[1])
            print("NuralWMSE Day1:%f\n" % wmse[0])
            print("NuralWMSE Day2:%f\n" % wmse[1])
            print("ZERO day1:%f\n" % zmse[0])
            print("ZERO day2:%f\n" % zmse[1])
            plt.plot(graphx,graphy)
            plt.show()

            """
            data=np.genfromtxt("../mycsv/mytrain2.0.1.csv",skip_header=1,delimiter=',',dtype='float32')
            predict = sess.run(y2, feed_dict={x: data[:, 26:147]})
            f=open("submission2.csv","w")
            f.write("Id,Predicted\n")
            for i in range(len(predict)):
                for j in range(62):
                    f.write(str(i+1)+"_"+str(j+1)+",")
                    if j==60:
                        f.write(str(predict[i,0])+"\n")
                        #f.write("0\n")
                    elif j==61:
                        f.write(str(predict[i,1])+"\n")
                    else:
                        f.write("0\n")
            f.close()
            """
    def nuralnet2(self):
        data=np.genfromtxt("../mycsv/special.csv",dtype="float32",delimiter=",",skip_header=1)
        x = tf.placeholder("float", shape=[None, 28])
        y_ = tf.placeholder("float", shape=[None, 2])
        with tf.name_scope('conv1') as scope:
            W = tf.Variable(tf.zeros([28,2])) # 正規分布でウェイトを初期化
            b = tf.Variable(tf.zeros([2]))
            y = tf.matmul(x, W) + b

        loss = tf.reduce_mean(tf.square(y - y_))
        #optimizer = tf.train.GradientDescentOptimizer(0.001)
        #optimizer = tf.train.FtrlOptimizer(0.01)
        #optimizer = tf.train.AdagradOptimizer(0.01)
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(loss)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        #Graph Data
        tf.scalar_summary('LOSS',loss)
        #tf.histogram_summary('Y',y)
        summary_op=tf.merge_all_summaries()
        summary_writer=tf.train.SummaryWriter('./',graph_def=sess.graph_def)

        print("反復開始")
        graphx=[]
        graphy=[]
        for step in range(10000):
            tmp = random.sample(range(len(data)), 1000)
            tmp = data[tmp, :]
            x_tmp = tmp[:, 0:28]
            y_tmp = tmp[:, 28:30]
            sess.run(train, feed_dict={x: x_tmp, y_: y_tmp})
            if step%1000==0:
                print(step,sess.run(W),sess.run(b))
                tf.scalar_summary('LOSS',loss)
                graphx.append(step)
                graphy.append(self.RMSET(sess.run(y, feed_dict={x: data[:, 0:28]})))
            summary_str=sess.run(summary_op,feed_dict={x: x_tmp, y_: y_tmp})
            summary_writer.add_summary(summary_str,step)

        predict = sess.run(y, feed_dict={x: data[:, 0:28]})
        rmse = [0., 0.]
        wmse = [0., 0.]

        for i in range(len(data)):
            for j in range(2):
                rmse[j] += pow(self.data[i, j + 207] - predict[i, j], 2)
                wmse[j] += self.data[i, 210] * abs(self.data[i, j + 207] - predict[i, j])
        rmse[0] = pow(rmse[0] / len(self.data), 0.5)
        rmse[1] = pow(rmse[1] / len(self.data), 0.5)
        wmse[0] = wmse[0] / len(self.data)
        wmse[1] = wmse[1] / len(self.data)
        print("NuralRMSE Day1:%f\n" % rmse[0])
        print("NuralRMSE Day2:%f\n" % rmse[1])
        print("NuralWMSE Day1:%f\n" % wmse[0])
        print("NuralWMSE Day2:%f\n" % wmse[1])
        plt.plot(graphx,graphy)
        plt.show()
    def RNN(self):
        x = tf.placeholder("float", shape=[None, 121])
        y_ = tf.placeholder("float", shape=[None, 60])
        W1 = tf.Variable(tf.truncated_normal([121,10],mean=2,stddev=1)) # 正規分布でウェイトを初期化
        W2 = tf.Variable(tf.truncated_normal([10,60],mean=2,stddev=1))
        b1 = tf.Variable(tf.zeros([10]))
        b2 = tf.Variable(tf.zeros([60]))
        y = tf.matmul(x, W1) + b1
        y2 = tf.matmul(y, W2) + b2
        loss = tf.reduce_mean(tf.square(y2 - y_))
        #optimizer = tf.train.GradientDescentOptimizer(0.01)
        #optimizer = tf.train.FtrlOptimizer(0.01)
        #optimizer = tf.train.AdagradOptimizer(0.01)
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(loss)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        #Graph Data
        tf.scalar_summary('LOSS',loss)
        #tf.histogram_summary('Y',y)
        summary_op=tf.merge_all_summaries()
        summary_writer=tf.train.SummaryWriter('./',graph_def=sess.graph_def)

        print("反復開始")
        graphx=[]
        graphy=[]
        for step in range(50001):
            tmp = random.sample(range(len(self.data)), 500)
            tmp = self.data[tmp, :]
            x_tmp = tmp[:, 26:147]
            y_tmp = tmp[:, 147:207]
            if step == 0:
                print("初期値\n",sess.run(W1),sess.run(b1))
            sess.run(train, feed_dict={x: x_tmp, y_: y_tmp})
            if step%1000==0:
                print(step,sess.run(W1),sess.run(b1))
                tf.scalar_summary('LOSS',loss)
                graphx.append(step)
                graphy.append(self.RMSET(sess.run(y, feed_dict={x: self.data[:, 26:147]})))
            summary_str=sess.run(summary_op,feed_dict={x: x_tmp, y_: y_tmp})
            summary_writer.add_summary(summary_str,step)

        predict = sess.run(y2, feed_dict={x: self.data[:, 26:147]})
        rmse = np.zeros([60])
        wmse = np.zeros([60])
        zmse = np.zeros([60])

        for i in range(len(predict)):
            for j in range(60):
                rmse[j] += pow(self.data[i, j + 146] - predict[i, j], 2)
                zmse[j] += self.data[i, 209] * abs(self.data[i, j + 146])
                wmse[j] += self.data[i, 209] * abs(self.data[i, j + 146] - predict[i, j])
        rmse[0] = pow(rmse[0] / len(predict), 0.5)
        rmse[1] = pow(rmse[1] / len(predict), 0.5)
        wmse[0] = wmse[0] / len(predict)
        wmse[1] = wmse[1] / len(predict)
        zmse[0] = zmse[0] / len(predict)
        zmse[1] = zmse[1] / len(predict)
        print("NuralRMSE Day1:%f\n" % rmse[0])
        #print("NuralRMSE Day2:%f\n" % rmse[1])
        print("NuralWMSE Day1:%f\n" % wmse[0])
        #print("NuralWMSE Day2:%f\n" % wmse[1])
        print("ZERO 1:%f\n" % zmse[0])
        #print("ZERO day2:%f\n" % zmse[1])
        plt.plot(graphx,graphy)
        plt.show()
    def Sigmoid(self,gosa): #０、０以上、０以下で場合分けして正確性をみる
        newdata=np.zeros([len(self.data),6])
        for i in range(len(self.data)):
            for j in [0]:
                if abs(self.data[i,j+207]) < gosa/2:
                    newdata[i,0]=1
                elif self.data[i,j+207]>0 and self.data[i,j+207]<gosa:
                    newdata[i,1] = 1
                elif self.data[i,j+207]<0 and self.data[i,j+207]>-gosa:
                    newdata[i,2] = 1
                elif self.data[i,j+207]>0:
                    newdata[i,3] = 1
                else:
                    newdata[i,4]=1
        x = tf.placeholder("float", shape=[None, 121])
        y_ = tf.placeholder("float", shape=[None, 6])
        W1 = tf.Variable(tf.truncated_normal([121,40],stddev=0.1)) # 正規分布でウェイトを初期化
        W2 = tf.Variable(tf.truncated_normal([40,6],stddev=0.1))
        b1 = tf.Variable(tf.zeros([40]))
        b2 = tf.Variable(tf.zeros([6]))
        y = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
        y2 = tf.nn.softmax(tf.matmul(y, W2) + b2)

        correct_prediction = tf.equal(tf.arg_max(y2,1),tf.arg_max(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

        loss = -tf.reduce_sum(y_*tf.log(y2))
        #optimizer = tf.train.GradientDescentOptimizer(0.2)
        #optimizer = tf.train.FtrlOptimizer(0.5)
        #optimizer = tf.train.AdagradOptimizer(0.01)
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(loss)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        with sess.as_default():
            print("反復開始")
            graphx=[]
            graphy=[]
            for step in range(20001):
                tmp = random.sample(range(len(self.data)), 100)
                newdata_tmp = newdata[tmp,:]
                tmp = self.data[tmp, :]
                x_tmp = tmp[:, 26:147]
                y_tmp = newdata_tmp[:, 0:6]
                sess.run(train, feed_dict={x: x_tmp, y_: y_tmp})
                if step%1000==0:
                    print(step)
                    graphx.append(step)
                    graphy.append(accuracy.eval(feed_dict={x: x_tmp, y_: y_tmp}))


            correct_prediction = tf.equal(tf.arg_max(y2,1),tf.arg_max(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
            print(sess.run(accuracy, feed_dict={x: self.data[:, 26:147],y_:newdata[:, 0:6]})*100)

            predict = sess.run(y2, feed_dict={x: self.data[:, 26:147]})
            wmse=0
            for i in range(len(self.data)):
                if np.argmax(predict[i,:])==1:
                    num=gosa*3/4
                elif np.argmax(predict[i,:])==2:
                    num=-1*gosa*3/4
                elif np.argmax(predict[i,:])==3:
                    num=gosa
                elif np.argmax(predict[i,:])==4:
                    num=-1*gosa
                else:
                    num=0
                wmse += self.data[i, 210] * abs(self.data[i, j + 207] - num)
            wmse/=len(self.data)
            print("WMSE:%f" %wmse)


            data=np.genfromtxt("../mycsv/mytrain2.0.1.csv",skip_header=1,delimiter=',',dtype='float32')
            predict = sess.run(y2, feed_dict={x: data[:, 26:147]})

            """
            predata=[]
            f=open("submission1.csv","r")
            f.readline()
            a=f.readlines()
            f.close()
            for i in range(120000):
                i=a[60+i*62].split(",")
                predata.append(i[1])
            """


            f=open("submission0.05.csv","w")
            f.write("Id,Predicted\n")
            for i in range(len(predict)):
                for j in range(62):
                    f.write(str(i+1)+"_"+str(j+1)+",")
                    if j==60:
                        flag=np.argmax(predict[i,0:3])
                        if flag==0:
                            f.write("0\n")
                        elif flag==1:
                            f.write(str(gosa)+"\n")
                        else:
                            f.write(str(-1*gosa)+"\n")
                            """
                    elif j==60:
                        f.write(str(predata[i]))
                        """
                    else:
                        f.write("0\n")
            f.close()



            plt.plot(graphx,graphy)
            plt.show()
    def Sigmoid2(self,gosa):
        newdata=np.zeros([len(self.data),2])
        for i in range(len(self.data)):
            for j in [0]:
                if abs(self.data[i,j+207]) < gosa:
                    newdata[i,0]=1
                else:
                    newdata[i,1]=1
        x = tf.placeholder("float", shape=[None, 121])
        y_ = tf.placeholder("float", shape=[None, 2])
        y_2 = tf.placeholder("float", shape=[None, 2])
        W1 = tf.Variable(tf.truncated_normal([121,40],stddev=0.1)) # 正規分布でウェイトを初期化
        W2 = tf.Variable(tf.truncated_normal([40,2],stddev=0.1))
        b1 = tf.Variable(tf.zeros([40]))
        b2 = tf.Variable(tf.zeros([2]))
        y = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
        y2 = tf.nn.softmax(tf.matmul(y, W2) + b2)

        W_2=tf.Variable(tf.truncated_normal([121,2],stddev=0.1))
        b_2 = tf.Variable(tf.zeros([2]))
        y_22=tf.matmul(x,W_2)+b_2

        loss = -tf.reduce_sum(y_*tf.log(y2))
        loss_2 = tf.reduce_mean(tf.square(y_22-y_2))
        #optimizer = tf.train.GradientDescentOptimizer(0.2)
        #optimizer = tf.train.FtrlOptimizer(0.5)
        #optimizer = tf.train.AdagradOptimizer(0.01)
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(loss)
        train2 = optimizer.minimize(loss_2)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        sess2=tf.Session()
        sess2.run(init)

        print("反復開始")
        graphx=[]
        graphy=[]
        for step in range(20001):
            tmp = random.sample(range(len(self.data)), 100)
            newdata_tmp = newdata[tmp,:]
            tmp = self.data[tmp, :]
            x_tmp = tmp[:, 26:147]
            y_tmp = newdata_tmp
            sess.run(train, feed_dict={x: x_tmp, y_: y_tmp})

        for step in range(20001):
            tmp = random.sample(range(len(self.data)), 100)
            tmp = self.data[tmp, :]
            x_tmp = tmp[:, 26:147]
            y_tmp = tmp[:, 207:209]
            sess2.run(train2, feed_dict={x: x_tmp, y_2: y_tmp})

        predict = sess.run(y2, feed_dict={x: self.data[:, 26:147]})
        wmse=0
        for i in range(len(self.data)):
            if np.argmax(predict[i,:])==1:
                num=sess2.run(y_22,feed_dict={x: self.data[i,26:147]})[0]
            else:
                num=0
            wmse += self.data[i, 210] * abs(self.data[i, j + 207] - num)
        wmse/=len(self.data)
        print("WMSE:%f" %wmse)

        """
        data=np.genfromtxt("../mycsv/mytrain2.0.1.csv",skip_header=1,delimiter=',',dtype='float32')
        predict = sess.run(y2, feed_dict={x: data[:, 26:147]})


        predata=[]
        f=open("submission1.csv","r")
        f.readline()
        a=f.readlines()
        f.close()
        for i in range(120000):
            i=a[60+i*62].split(",")
            predata.append(i[1])


        f=open("submission0.05.csv","w")
        f.write("Id,Predicted\n")
        for i in range(len(predict)):
            for j in range(62):
                f.write(str(i+1)+"_"+str(j+1)+",")
                if j==61:
                    flag=np.argmax(predict[i,0:3])
                    if flag==0:
                        f.write("0\n")
                    elif flag==1:
                        f.write(str(gosa)+"\n")
                    else:
                        f.write(str(-1*gosa)+"\n")
                elif j==60:
                    f.write(str(predata[i]))
                else:
                    f.write("0\n")
        f.close()
        """


        plt.plot(graphx,graphy)
        plt.show()
    def RMSET(self,predict):
        rmse = [0., 0.]
        wmse = [0., 0.]
        for i in range(len(self.data)):
            for j in range(2):
                rmse[j] += pow(self.data[i, j + 207] - predict[i, j], 2)
                wmse[j] += self.data[i, 210] * abs(self.data[i, j + 207] - predict[i, j])
        rmse[0] = pow(rmse[0] / len(self.data), 0.5)
        rmse[1] = pow(rmse[1] / len(self.data), 0.5)
        wmse[0] = wmse[0] / len(self.data)
        wmse[1] = wmse[1] / len(self.data)
        return rmse[0]
    def Final(self,gosa): #０、０以上、０以下で場合分けして正確性をみる
        trainx=self.data[:,26:147]
        trainy=self.data[:,207]
        testx=self.data[:,26:147]
        testy=self.data[:,207]
        testw=self.data[:,210]
        newdata=np.zeros([len(trainy),3])
        mu_tmp=[0]*25
        hensa=[0]*25
        for i in range(25):
            mu_tmp[i]=np.average(self.data[:,i+1])
            for j in range(len(self.data)):
                hensa[i]+=pow(self.data[j,i+1]-mu_tmp[i],2)
            hensa[i]/=len(self.data)
            hensa[i]=pow(hensa[i],0.5)
            for j in range(len(self.data)):
                

        for i in range(len(trainy)):
            if abs(trainy[i]) < gosa:
                newdata[i,0]=1
            elif trainy[i]>0:
                newdata[i,1] = 1
            else:
                newdata[i,2] = 1

        #回帰のためのデータ作成部分
        plusdata=[]
        minusdata=[]
        for i in range(len(trainy)):
            if trainy[i]>=gosa:
                plusdata.append(i)
            elif trainy[i]<=-gosa:
                minusdata.append(i)
        plusdatax = trainx[plusdata]
        plusdatay = trainy[plusdata]
        minusdatax = trainx[minusdata]
        minusdatay = trainy[minusdata]

        optimizer = tf.train.AdamOptimizer()

        #plusのときの回帰
        px = tf.placeholder("float", shape=[None, 121])
        py_ = tf.placeholder("float", shape=[None])
        pW = tf.Variable(tf.truncated_normal([121,1],stddev=0.1))
        pb = tf.Variable(tf.truncated_normal([1],stddev=0.1))
        py = tf.matmul(px, pW) + pb
        ploss = tf.reduce_mean(tf.square(py_ - py))
        ptrain = optimizer.minimize(ploss)

        #minusのときの回帰
        mx = tf.placeholder("float", shape=[None, 121])
        my_ = tf.placeholder("float", shape=[None])
        mW = tf.Variable(tf.truncated_normal([121,1],stddev=0.1))
        mb = tf.Variable(tf.truncated_normal([1],stddev=0.1))
        my = tf.matmul(mx, mW) + mb
        mloss = tf.reduce_mean(tf.square(my_ - my))
        mtrain = optimizer.minimize(mloss)

        x = tf.placeholder("float", shape=[None, 146])
        y_ = tf.placeholder("float", shape=[None,3])
        W1 = tf.Variable(tf.truncated_normal([146,40],stddev=0.1)) # 正規分布でウェイトを初期化
        W2 = tf.Variable(tf.truncated_normal([40,3],stddev=0.1))
        b1 = tf.Variable(tf.zeros([40]))
        b2 = tf.Variable(tf.zeros([3]))
        y = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
        y2 = tf.nn.softmax(tf.matmul(y, W2) + b2)

        loss = -tf.reduce_sum(y_*tf.log(y2))
        train = optimizer.minimize(loss)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        sessp = tf.Session()
        sessp.run(init)
        sessm = tf.Session()
        sessm.run(init)
        with sess.as_default():
            #０判別ニューラルネット
            print("0Nural")
            for step in range(20001):
                tmp = random.sample(range(len(self.data)), 100)
                x_tmp = self.data[tmp, 1:147]
                y_tmp = newdata[tmp,:]
                sess.run(train, feed_dict={x: x_tmp, y_: y_tmp})

        print("PlusReg")
        with sessp.as_default():
            #プラス回帰ネット
            for step in range(5000):
                tmp = random.sample(range(len(plusdatax)), 100)
                x_tmp = plusdatax[tmp]
                y_tmp = plusdatay[tmp]
                sessp.run(ptrain, feed_dict={px: x_tmp, py_: y_tmp})

        print("MinusReg")
        with sessm.as_default():
            #プラス回帰ネット
            for step in range(5000):
                tmp = random.sample(range(len(minusdatax)), 100)
                x_tmp = minusdatax[tmp]
                y_tmp = minusdatay[tmp]
                sessm.run(mtrain, feed_dict={mx: x_tmp, my_: y_tmp})

        print("Finished Reg")
        predict = sess.run(y2, feed_dict={x: self.data[:,1:147]})
        wmse=0
        for i in range(len(testx)):
            if np.argmax(predict[i,:])==0:
                num=0
            elif np.argmax(predict[i,:])==1:
                num=sessp.run(py, feed_dict={px: testx[i:]})[0]
                if num<gosa:
                    num=gosa
            else:
                num=sessm.run(my, feed_dict={mx: testx[i:]})[0]
                if num>-gosa:
                    num=-gosa
            wmse += testw[i] * abs(testy[i] - num)
        wmse/=len(testx)
        print("WMSE:%f" %wmse)


        data=np.genfromtxt("../mycsv/mytrain2.0.1.csv",skip_header=1,delimiter=',',dtype='float32')
        predict = sess.run(y2, feed_dict={x: data[:, 1:147]})

        """
        predata=[]
        f=open("submission1.csv","r")
        f.readline()
        a=f.readlines()
        f.close()
        for i in range(120000):
            i=a[60+i*62].split(",")
        """

        f=open("submission0.051.csv","w")
        f.write("Id,Predicted\n")
        for i in range(len(predict)):
            for j in range(62):
                f.write(str(i+1)+"_"+str(j+1)+",")
                if j==60:
                    flag=np.argmax(predict[i,0:3])
                    if flag==0:
                        f.write("0\n")
                    elif flag==1:
                        num=sessp.run(py, feed_dict={px: data[i:, 26:147]})[0]
                        num=num[0]
                        if num<gosa:
                            num=gosa
                        f.write(str(num)+"\n")
                    else:
                        num=sessm.run(my, feed_dict={mx: data[i:, 26:147]})[0]
                        num=num[0]
                        if num>-gosa:
                            num=-gosa
                        f.write(str(num)+"\n")
                else:
                    f.write("0\n")
        f.close()
    def cross_validation(self,f,x_data,y_data,w_data,times=10):
        randombox=random.sample(len(x_data),len(x_data))
        newx=x_data[randombox,:]
        newy=y_data[randombox,:]
        neww=w_data[randombox,:]

        returnnum=0.
        for i in range(times):
            print("%d of %d" % (i,times))
            crossnum=np.array([False]*len(x_data))
            crossnum[i*len(x_data)/10:(i+1)*len(x_data)/10]=True
            returnnum+=f(newx[~crossnum,:],newy[~crossnum,:],newx[crossnum,:],newy[crossnum,:],neww[crossnum,:])
        return returnnum/times


if __name__=="__main__":
    myclass=DataCreate(1,2,1,0)
    myclass.Final(gosa=0.02)
    myclass.printTime()

