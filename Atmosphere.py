
'''
Copyright (c) 2019 Seiji Arthur Murakami. All Rights Reserved.

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


This program is based in below paper

- U.S. Standard Atmosphere, 1976

'''

from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np

class Atmosphere:

    def __init__(self):

        self.r0 = 6356.7660 #Earth radius in [km]
        self.g0 = 9.80665 #Gravity at surface [m/s2]
        self.T0 = 288.15  #Temperature at surface [K]
        self.P0 = 101325.0  #Pressure at surface [Pa][N/m2]
        self.rho0 = 1.225 #Air Density at surface [kg/m3]
        self.a0 = 340.29  #Sound speed at surface [m/s]
        self.u0 = 1.7894 / 100000 #Air static viscosity [Ns/m2]
        self.v0 = 1.4607 / 100000 #Air dynamic viscosity [m2/s]
        self.M0 = 28.9644  #molecular weight agerage [kg/kmol]
        self.Rstar = 8.31432 * 1000 #Gas constant [Nm/kmolK]
        self.NA = 6.022169e+26 #Avogadro's constant [kmol-1]
        self.gamma = 2.4  #gas specific heat ratio, unitless.
        self.S = 110 #Sutherland's constant [K]
        self.beta = 1.458e-6 # quantity constant [smk**0.5]

        self.Hb = [0.0,11.0,20.0,32.0,47.0,51.0,71.0,84.8520]
        self.Lmb = [-6.5,0.0,1.0,2.8,0.0,-2.8,-2.0]
        self.Tmb = []
        self.Pb = []
        self.Tmb.append(self.T0)
        self.Pb.append(self.P0)

        for i in range(1,8):
            self.Tmb.append(self.Tmb[i-1] + self.Lmb[i-1] * (self.Hb[i]-self.Hb[i-1]))

            if i == 2 or i == 5:
                self.Pb.append(self.Pb[i-1]*np.exp((-self.g0 * self.M0 * (self.Hb[i] - self.Hb[i-1]))/(self.Rstar/1000*self.Tmb[i-1])))

            else:
                self.Pb.append(self.Pb[i-1]*(self.Tmb[i-1]/self.Tmb[i])**((self.g0 * self.M0)/(self.Rstar/1000 * self.Lmb[i-1])))

            
    def geo_H_conv(self,altitude):

        return (self.r0*altitude)/(self.r0 + altitude)

    def g_alt(self,altitude):

        return( self.g0 * (self.r0/(self.r0 + altitude)) ** 2)

    def ang_accel(self,altitude):

        w = 2 * np.pi / 86164.091

        return((self.r0 +altitude)*1000*w*w)


    def temperature(self, altitude):

        geo_alt = self.geo_H_conv(altitude)


        if altitude < 86 :
            for i in range(0,7): 
                if geo_alt < self.Hb[i+1] :
                    return (self.Tmb[i] + self.Lmb[i]*(geo_alt - self.Hb[i]))
        elif altitude <= 91:
            return 186.8673

        elif altitude <= 110:
            return(263.1905 + (-76.3232)*(1-((altitude - 91)/(-19.9429)) ** 2) ** 0.5)

        elif altitude <= 120:
            return(240 + 12*(altitude - 110))

        elif altitude <= 1000:

            Tinf = 1000
            T10 = 360
            Z10 = 120
            lamda = 12/(Tinf - T10)
            gsy = (altitude - Z10)*(self.r0 + Z10)/(self.r0 + altitude)

            return(Tinf - (Tinf - T10)*np.exp(-lamda*gsy))

        elif altitude > 1000:

            return(1000)


    def pressure(self, altitude):
        
        geo_alt = self.geo_H_conv(altitude)

        Ztableupper = np.array([86.0, 87.0, 88.0, 89.0, 90.0, 91.0,
                93.0, 95.0, 97.0, 99.0, 101.0, 103.0, 105.0, 107.0,
                109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0,
                116.0, 117.0, 118.0, 119.0, 120.0, 125.0, 130.0,
                135.0, 140.0, 145.0, 150.0, 160.0, 170.0, 180.0,
                190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0,
                260.0, 270.0, 280.0, 290.0, 300.0, 310.0, 320.0,
                330.0, 340.0, 350.0, 360.0, 370.0, 380.0, 390.0,
                400.0, 410.0, 420.0, 430.0, 440.0, 450.0, 460.0,
                470.0, 480.0, 490.0, 500.0, 525.0, 550.0, 575.0,
                600.0, 625.0, 650.0, 675.0, 700.0, 725.0, 750.0,
                775.0, 800.0, 825.0, 850.0, 875.0, 900.0, 925.0,
                950.0, 975.0, 980.0, 990.0, 1000.0]) # (m)
        
        Ptableupper = np.array([3.7338E-1, 3.1259E-1, 2.6173E-1, 2.1919E-1, 1.8359E-1,
                1.5381E-1, 1.0801E-1, 7.5966E-2, 5.3571E-2, 3.7948E-2, 2.7192E-2,
                1.9742E-2, 1.4477E-2, 1.0751E-2, 8.1142E-3, 7.1042E-3, 6.2614E-3,
                5.5547E-3, 4.9570E-3, 4.4473E-3, 4.0096E-3, 3.6312E-3, 3.3022E-3,
                3.0144E-3, 2.7615E-3, 2.5382E-3, 1.7354E-3, 1.2505E-3, 9.3568E-4,
                7.2028E-4, 5.6691E-4, 4.5422E-4, 3.0395E-4, 2.1210E-4, 1.5271E-4,
                1.1266E-4, 8.4736E-5, 6.4756E-5, 5.0149E-5, 3.9276E-5, 3.1059E-5,
                2.4767E-5, 1.9894E-5, 1.6083E-5, 1.3076E-5, 1.0683E-5, 8.7704E-6,
                7.2285E-6, 5.9796E-6, 4.9630E-6, 4.1320E-6, 3.4498E-6, 2.8878E-6,
                2.4234E-6, 2.0384E-6, 1.7184E-6, 1.4518E-6, 1.2291E-6, 1.0427E-6,
                8.8645E-7, 7.5517E-7, 6.4468E-7, 5.5155E-7, 4.7292E-7, 4.0642E-7,
                3.5011E-7, 3.0236E-7, 2.1200E-7, 1.5137E-7, 1.1028E-7, 8.2130E-8,
                6.2601E-8, 4.8865E-8, 3.9048E-8, 3.1908E-8, 2.6611E-8, 2.2599E-8,
                1.9493E-8, 1.7036E-8, 1.5051E-8, 1.3415E-8, 1.2043E-8, 1.0873E-8,
                9.8635E-9, 8.9816E-9, 8.2043E-9, 8.0597E-9, 7.7805E-9, 7.5138E-9]) # (Pa)


        if altitude < 86 :
            
            for i in range(0,7):
                if geo_alt <= self.Hb[i+1]:
                    break

            if (i == 1 or i == 4):
                return(self.Pb[i]*np.exp((-self.g0 * self.M0 * (geo_alt - self.Hb[i]))/(self.Rstar/1000*self.Tmb[i])))
            
            else:
                return(self.Pb[i]*(self.Tmb[i]/self.temperature(altitude))**((self.g0 * self.M0)/(self.Rstar/1000 * self.Lmb[i])))
        elif altitude <= 1000:

            f = interpolate.interp1d(Ztableupper,Ptableupper,kind = 'cubic')

            return(f(altitude))
        
    def density(self, altitude):

        return((self.pressure(altitude)*self.M0)/(self.Rstar * self.temperature(altitude)))

    def sound(self, altitude):

        return((self.gamma * self.Rstar * self.temperature(altitude) / self.M0) ** 0.5)

    def viscosity_dynamic(self, altitude):

        return((self.beta * self.temperature(altitude) ** 1.5) / (self.temperature(altitude) + self.S))

    def viscosity_kinetic(self,altitude):

        return(self.viscosity_dynamic(altitude)/self.density(altitude))


if __name__ == '__main__':
    
    print('Altitude Air value calculation')

    atmosphere = Atmosphere()

    km = []
    temp = []
    pressure = []
    density = []
    sound = []
    viscosity_dynamic = []
    viscosity_kinetic = []
    gravity = []
    ang_accel = []

    plt.figure()

    for i in range(1000):
        km.append(i)
#        temp.append(atmosphere.temperature(i))
#        gravity.append(atmosphere.g_alt(i))
#        ang_accel.append(atmosphere.ang_accel(i))
#        pressure.append(atmosphere.pressure(i))
#        density.append(atmosphere.density(i))
#        sound.append(atmosphere.sound(i))
#        viscosity_dynamic.append(atmosphere.viscosity_dynamic(i) * 100000)
        viscosity_kinetic.append(atmosphere.viscosity_kinetic(i))

        print(i,atmosphere.viscosity_kinetic(i))


    plt.plot(viscosity_kinetic,km)
    plt.xlim(1,10e10)
    plt.ylim(0,1000)
    plt.xscale('log')
    plt.xlabel('Kinetic Viscosity(m2/s)')
    plt.ylabel('Altitude(km)')
    plt.grid()
    plt.title('Kinetic Viscosity plot')
    plt.legend(loc='best', fontsize=12)
    plt.savefig("Altitude VS Kinetic Viscosity.png")
    plt.close()


