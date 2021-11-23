import numpy as np
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
from itertools import permutations


#__author__ = Laura Kriener

class YinYangDataset(Dataset):
    def __init__(self, r_small=0.2, r_big=1., bottom_left=0.0, top_right=1., size=1000, seed=42, multiply_input_layer=1, flipped_coords=False):
        # calculations copied from
        # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
        assert type(multiply_input_layer) == int
        if seed is not None:
            np.random.seed(seed)
        self.r_small = r_small
        self.r_big = r_big
        self.__vals = []
        self.__cs = []
        self.class_names = ['yin', 'yang', 'dot']
        for i in range(size):
            # keep num of class instances balanced
            goal = np.random.randint(3)
            x, y, c = self.get_sample(goal=goal)
            # x, y in range 0 to 1 -> adjust
            x = bottom_left + x * (top_right - bottom_left)
            x_flipped = top_right - x + bottom_left
            y = bottom_left + y * (top_right - bottom_left)
            y_flipped = top_right - y + bottom_left
            val = []
            for i in range(multiply_input_layer):
                val.append(x)
                val.append(y)
                if flipped_coords:
                    val.append(x_flipped)
                    val.append(y_flipped)
            self.__vals.append(np.array(val))
            self.__cs.append(c)
        self.__vals = np.array(self.__vals)
        self.__cs = np.array(self.__cs)

    def d_plus(self, x, y):
        return np.sqrt((x - 0.5*self.r_big)**2 + y**2)

    def d_minus(self, x, y):
        return np.sqrt((x + 0.5*self.r_big)**2 + y**2)

    def yin_yang(self, x, y):
        dplus = self.d_plus(x, y)
        dminus = self.d_minus(x, y)
        criterion1 = dplus <= self.r_small
        criterion2 = dminus > self.r_small and dminus <= 0.5 * self.r_big
        criterion3 = y > 0 and dplus > 0.5 * self.r_big
        yin = criterion1 or criterion2 or criterion3
        circles = dplus < self.r_small or dminus < self.r_small
        if circles:
            return 2
        return int(yin)

    def get_sample(self, goal=None):
        x = np.random.rand()*2. - 1.
        y = np.random.rand()*2. - 1.
        while np.sqrt(x**2 + y**2) > 1:
            x = np.random.rand()*2. - 1.
            y = np.random.rand()*2. - 1.
        c = self.yin_yang(x, y)
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        if goal is None:
            return x, y, c
        elif goal == c:
            return x, y, c
        else:
            x, y, c = self.get_sample(goal)
            return x, y, c

    def __getitem__(self, index):
        return self.__vals[index], self.__cs[index]

    def __len__(self):
        return len(self.__cs)



class SineDataset(Dataset):
    def __init__(self, wavelength=1.0, amplitude=0.2, bottom_left=0.0, top_right=1., size=1000, seed=42, flipped_coords=False):
        if seed is not None:
            np.random.seed(seed)
        self.wavelen = wavelength
        self.ampl = amplitude
        self.__vals = []
        self.__cs = []
        self.class_names = ['top', 'bottom']
        for i in range(size):
            # keep num of class instances balanced
            goal = np.random.randint(2)
            x, y, c = self.get_sample(goal=goal)
            # x, y in range 0 to 1 -> adjust
            x = bottom_left + x * (top_right - bottom_left)
            x_flipped = top_right - x + bottom_left
            y = bottom_left + y * (top_right - bottom_left)
            y_flipped = top_right - y + bottom_left
            val = []
            val.append(x)
            val.append(y)
            if flipped_coords:
                val.append(x_flipped)
                val.append(y_flipped)
            self.__vals.append(np.array(val))
            self.__cs.append(c)
        self.__vals = np.array(self.__vals)
        self.__cs = np.array(self.__cs)

    def get_sample(self, goal=None):
        x = np.random.rand()
        y = np.random.rand()
        c = y > np.sin(2*np.pi*x/self.wavelen)*self.ampl + 0.5
        if goal is None:
            return x, y, c
        elif goal == c:
            return x, y, c
        else:
            x, y, c = self.get_sample(goal)
            return x, y, c

    def __getitem__(self, index):
        return self.__vals[index], self.__cs[index]

    def __len__(self):
        return len(self.__cs)


class BarsDataset(Dataset):
    def __init__(self, square_size, bottom_left=0.0, top_right=1.0, noise_level=1e-2, samples_per_class=10, seed=42):
        if seed is not None:
            np.random.seed(seed)
        debug = False
        self.__vals = []
        self.__cs = []
        self.class_names = ['horiz', 'vert', 'diag']
        ones = list(np.ones(square_size) + (top_right - 1.))
        if debug:
            print(ones)
        starter = [ones]
        for i in range(square_size - 1):
            starter.append(list(np.zeros(square_size) + bottom_left))
        if debug:
            print('Starter')
            print(starter)
        horizontals = []
        for h in permutations(starter):
            horizontals.append(list(h))
        horizontals = np.unique(np.array(horizontals), axis=0)
        if debug:
            print('Horizontals')
            print(horizontals)
        verticals = []
        for h in horizontals:
            v = np.transpose(h)
            verticals.append(v)
        verticals = np.array(verticals)
        if debug:
            print('Verticals')
            print(verticals)
        diag = [top_right - bottom_left for i in range(square_size)]
        first = np.diag(diag) + bottom_left
        second = first[::-1]
        diagonals = [first, second]
        if debug:
            print('Diagonals')
            print(diagonals)
        n = 0
        idx = 0
        while n < samples_per_class:
            h = horizontals[idx].flatten()
            h = list(h + np.random.rand(len(h))*noise_level)
            self.__vals.append(h)
            self.__cs.append(0)
            n += 1
            idx += 1
            if idx >= len(horizontals):
                idx = 0
        n = 0
        idx = 0
        while n < samples_per_class:
            v = verticals[idx].flatten()
            v = list(v + np.random.rand(len(v))*noise_level)
            self.__vals.append(v)
            self.__cs.append(1)
            n += 1
            idx += 1
            if idx >= len(verticals):
                idx = 0
        n = 0
        idx = 0
        while n < samples_per_class:
            d = diagonals[idx].flatten()
            d = list(d + np.random.rand(len(d))*noise_level)
            self.__vals.append(d)
            self.__cs.append(2)
            n += 1
            idx += 1
            if idx >= len(diagonals):
                idx = 0

    def __getitem__(self, index):
        return np.array(self.__vals[index]), np.array(self.__cs[index])

    def __len__(self):
        return len(self.__cs)




def plot_yy(x, label, ax=None):
    if ax is None:
        ax = plt.gca()
    c1 = np.argwhere(label==0).flatten()
    c2 = np.argwhere(label==1).flatten()
    c3 = np.argwhere(label==2).flatten()
    ax.scatter(x[c1, 0], x[c1, 1], c="red")
    ax.scatter(x[c2, 0], x[c2, 1], c="blue")
    ax.scatter(x[c3, 0], x[c3, 1], c="green")
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_aspect(1)

def plot_sine(x, label, ax=None):
    if ax is None:
        ax = plt.gca()
    c1 = np.argwhere(label==0).flatten()
    c2 = np.argwhere(label==1).flatten()
    ax.scatter(x[c1, 0], x[c1, 1], c="red")
    ax.scatter(x[c2, 0], x[c2, 1], c="blue")
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_aspect(1)

def plot_bars(x, size):
    plt.imshow(x.reshape(size, size))

if __name__=="__main__":
    #test yin yang
    X, Y = YinYangDataset(size=1000)[:]
    plot_yy(X, Y)
    plt.title("Yin Yang Dataset")
    plt.show()

    #test bars
    s = 5
    n = 10
    X, Y = BarsDataset(s, samples_per_class=n, noise_level=0.1)[:]
    plt.figure(figsize=(15,6))
    plt.suptitle("Bars Dataset")
    for i in range(n):
        plt.subplot(3,n,i+1)
        plot_bars(X[np.argwhere(Y == 0).flatten()[i]], s)
    for i in range(n):
        plt.subplot(3, n, i + n + 1)
        plot_bars(X[np.argwhere(Y == 1).flatten()[i]], s)
    for i in range(n):
        plt.subplot(3,n,i + 2 * n + 1)
        plot_bars(X[np.argwhere(Y == 2).flatten()[i]], s)
    plt.show()

    # test sine
    X, Y = SineDataset(size=1000)[:]
    plot_sine(X, Y)
    plt.title("Sine Dataset")
    plt.show()

