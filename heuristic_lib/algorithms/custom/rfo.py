import random
import math
import numpy as np
from heuristic_lib.benchmarks.utility import Utility

__all__ = ['RedFoxOptimizationAlgorithm']


class RedFoxOptimizationAlgorithm: 
    
    def __init__(self, D, NP, nFES, benchmark):
        self.benchmark = Utility().get_benchmark(benchmark)
        self.D = D  # dimension of the problem
        self.NP = NP  # population size
        self.nFES = nFES  # number of function evaluations
        self.Index = [0] * self.NP
        self.Foxes = [[0 for _i in range(self.D)]
                          for _j in range(self.NP)]  # tilkiler
        self.Foxes_tmp = [[0 for _i in range(self.D)] for _j in range(self.NP)]  # geçici popülasyon
        self.Fitness = [0.0] * self.NP  # fitness değeri
        self.nbest = [0.0] * self.NP  # şimdiye kadarki en iyi çözüm (best value)
        self.Lower = self.benchmark.Lower  # Fonksiyon alt sınırı (lower bound)
        self.Upper = self.benchmark.Upper  # Fonksiyon üst sınırı (upper bound)
        self.BestFox = None  # En iyi üye (the best individual)
        self.SecondFox = None  # En iyi ikinci üye (the best individual)
        self.evaluations = 0
        self.eval_flag = True  # İterasyon bayrağı (evaluations flag)
        self.Fun = self.benchmark.function() #Fun ile cost fonksiyonunu tanımlıyoruz, yani benchmark'ı

    def init_rfo(self):
        """Red Fox popülasyonu başlatıcısı (Initialize red fox population)"""
        for i in range(self.NP):
            for j in range(self.D):
                self.Foxes[i][j] = random.uniform(0.0, 1.0) * (self.Upper - self.Lower) + self.Lower
            self.Fitness[i] = 1.0  # Fitness fonksiyonu sonuçlarının ilk değerleri
        #print("Fox'lar ilklendi")

    def eval_true(self):
        """Check evaluations."""

        if self.evaluations == self.nFES:
            self.eval_flag = False

    def sort_replace_rfo(self):  #
        """Bubble sort ile tilkileri sıralıyoruz """
        for i in range(self.NP):
            self.Index[i] = i
        
        for i in range(0, (self.NP - 1)):
            j = i + 1
            for j in range(j, self.NP):
                if self.Fitness[i] > self.Fitness[j]:
                    z = self.Fitness[i]  # exchange fitness
                    self.Fitness[i] = self.Fitness[j]
                    self.Fitness[j] = z
                    z = self.Index[i]  # exchange indexes
                    self.Index[i] = self.Index[j]
                    self.Index[j] = z
		
        for i in range(self.NP):
            for j in range(self.D):
                self.Foxes_tmp[i][j] = self.Foxes[i][j]
        for i in range(self.NP):
            for j in range(self.D):
                self.Foxes[i][j] = self.Foxes_tmp[self.Index[i]][j]
        #print("Tilkiler sıralandı")

    def FindLimits(self, k):
        """Eğer limitler dışına çıkanlar varsa, limitler içerisine alıyoruz"""
        for i in range(self.D):
            if self.Foxes[k][i] < self.Lower:
                self.Foxes[k][i] = self.Lower
            if self.Foxes[k][i] > self.Upper:
                self.Foxes[k][i] = self.Upper
        #print("Limitler düzeltildi")

    def move_global_rfo(self):
        """Red Fox'ları global fazda hareket ettirelim"""
        self.Foxes_tmp = self.Foxes
        fitness = self.Fitness
        
        for i in range(0,self.NP):
            dist = self.find_euclidian_distance(self.BestFox,self.Foxes[i])
            alpha = random.uniform(0,dist)
            for j in range(0,len(self.Foxes[i])):
                self.Foxes[i][j] = self.Foxes[i][j] + np.sign(self.BestFox[j] - self.Foxes[i][j]) * alpha
            
            self.FindLimits(i)
            self.Fitness[i] = self.Fun(self.D, self.Foxes[i])
            
            if self.Fitness[i] > fitness[i]: #Global Minimum için yapıldı, Global Max için tersine çevrilmeli
                self.Foxes[i] = self.Foxes_tmp[i] #Eğer yeni fitness değeri daha kötüyse eskisine dönüyoruz
                self.Fitness[i] = fitness[i]
        #print("Tilkiler global hareket etti")
        
    def move_local_rfo(self):
        """Red Fox'ları local fazda hareket ettirelim"""
        pi=np.pi
        sin=np.sin
        cos=np.cos
        random=np.random
        n=self.D
        if random.uniform(0,1) > 0.75:
            scaling_a = random.uniform(0,0.2)
            phi0 = random.uniform(0,2*pi)
            radius = random.uniform(0,1) #theta
            if phi0 != 0:
                radius = (sin(phi0)/(phi0)) * scaling_a
			
            temp = np.ndarray(n)
            phi_ = np.ndarray(n)
            for i in range(0,n):
                phi_[i] = np.random.uniform(0,2*pi)
            for i in range(0,n):
                tmp=0
                if i != 0:
                    for k in range(0,i):
                        tmp = tmp + scaling_a * radius * sin(phi_[k])
                if i != (n-1):
                    tmp = tmp + scaling_a * radius * cos(phi_[i])
                else:
                    tmp = tmp + scaling_a * radius * sin(phi_[i])
                temp[i] = tmp
			
            for i in range(0,n):
                self.Foxes[i] = self.Foxes[i] + temp
                self.FindLimits(i)
        #print("Tilkiler local hareketlendi")
			
    def find_euclidian_distance(self,fox1,fox2):
        dist = 0
        for i in range(0,len(fox1)):
            dist = dist + (fox1[i]-fox2[i])**2
        dist = dist**(1/2)
        #dist = dist**(1/2)
        #print("Öklid uzaklığı bulundu")
        return dist
	
    def reproduction_and_leaving_herd_rfo(self):
        habitat_center = self.BestFox
        for i in range(0,len(self.BestFox)):
            habitat_center[i] = (self.BestFox[i] + self.SecondFox[i])/2
        
        habitat_diameter = self.find_euclidian_distance(self.BestFox,self.SecondFox)
        FivePercent = math.ceil(self.NP*0.05)
        
        for i in range(1,FivePercent+1):
            K = random.uniform(0,1.0)
            if  K<0.45: #Reproduction of the alpha couple
                for j in range(0,len(habitat_center)):
                    self.Foxes[-i][j] = habitat_center[j] * K
                #print("Tilkiler öldü ve yeniden doğuyor")
            else: #New nomadic individual
                replaced_fox = self.Foxes[-i]
                for j in range(self.D):
                    replaced_fox[j] = random.uniform(0, 1) * (self.Upper - self.Lower) + self.Lower
                dist = self.find_euclidian_distance(replaced_fox,habitat_center)
                while dist < habitat_diameter:
                    for j in range(self.D):
                        replaced_fox[j] = random.uniform(0, 1) * (self.Upper - self.Lower) + self.Lower
                    dist = self.find_euclidian_distance(replaced_fox,habitat_center)
                self.Foxes[-i] = replaced_fox
                #print("Tilkiler göç ediyor")
		
		
    def run(self):
        """Run."""
        self.init_rfo()

        while self.eval_flag is not False:
		# evaluate new solutions
            for i in range(self.NP):
                self.eval_true()
                if self.eval_flag is not True:
                    break
                
                self.Fitness[i] = self.Fun(self.D, self.Foxes[i])
                self.evaluations = self.evaluations + 1
			
			# tilkileri fitness fonksiyonuna göre sıralıyoruz
            self.sort_replace_rfo()
			# en iyi üyeleri seçiyoruz
            self.BestFox = self.Foxes[0]  # En iyi üye (the best individual)
            self.SecondFox = self.Foxes[1]  # İkinci üye (the second individual)
            self.move_global_rfo()  #Global Search fazında hareket ettirme
            self.move_local_rfo()    #Local Search fazında hareket ettirme
            
			# tilkileri fitness fonksiyonuna göre yeniden sıralıyoruz
            self.sort_replace_rfo()
            self.BestFox = self.Foxes[0]  # En iyi üye (the best individual)
            self.SecondFox = self.Foxes[1]  # İkinci üye (the second individual)
            
            self.reproduction_and_leaving_herd_rfo()

        return self.Fitness[0]
