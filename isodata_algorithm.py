import numpy as np


class ISODATA:
    def __init__(self, x, z, K:int, gn: float, gs: float, gc: float, L:int, I:int):
        self.x = x
        self.z = z
        self.K = K
        self.gn = gn
        self.gs = gs
        self.gc = gc
        self.L = L
        self.I = I
        self.N = len(x)
        self.Nc = len(z)

    def start(self):
        iter0 = 0
        self.step2(iter0)

    def step2(self, iter):  # заданные точки x распределяются по кластерам, соответствующим выбранным исходным центрам
        iter1 = iter + 1
        self.S = [[] for _ in range(self.Nc)]
        for point in self.x:
            indx = np.argmin(np.linalg.norm(self.z - point, axis=1))
            self.S[indx].append(point)

        # Шаг 3
        # удаляются подмножества(кластеры), в которых менее gn элементы
        for j in range(self.Nc):
            if len(self.S[j]) < self.gn:
                self.S.remove(self.S[j])
                self.Nc = self.Nc - 1
        # Шаг 4
        # Меняем значения центров кластеров, приравнивая их выборочным средним для соответствующих кластеров
        self.z = [np.mean(self.S[j], axis=0) for j in range(self.Nc)]  # массив с координатами средних значений

        # Шаг 5
        # вычисляются средние расстояния между точками кластера и центром этого кластера
        self.D = [np.mean([np.linalg.norm(x - self.z[j]) for x in self.S[j]]) for j in range(self.Nc)]

        # Шаг 6
        # Вычисляется обобщённое среднее расстояние между точками в отдельных кластерах и соответствующими центрами кластеров
        self.D_common = 1 / self.N * sum([len(self.S[j]) * self.D[j] for j in range(self.Nc)])

        # Шаг 7
        # если итерация последняя переход к 11 шагу
        if iter1 == self.I:  # (а)
            self.gc = 0
            self.step11(iter1)

        # если количество центров кластеров меньше половины необходимого числа кластеров, то переходим к 8 шагу
        elif self.Nc <= self.K / 2:  # (б)
            self.step8(iter1)

        # если текщий цикл итерации чётный или количество центров кластеров в 2 раза больше необходимого числа кластеров, то переходим к 11 шагу
        elif ((iter1 % 2 == 0) or (self.Nc >= 2 * self.K)):
            self.step11(iter1)

        # если ничего не выполнилось, то возвращаемся к шагу 2
        else:
            self.step2(iter1)

    # Шаг 8 для каждого кластера вычисляется вектор СКО
    def step8(self, iter):
        iter2 = iter
        self.std = [np.std(cluster, axis=0) for cluster in self.S]

        # Шаг 9
        # список содержащий максимальную компоненту каждого из векторов СКО
        stdmax = [max(comp) for comp in self.std]
        stdmax_inds = [ar.argmax() for ar in self.std]  # список индексов максимальных компонент

        # Шаг 10
        Nc_old = self.Nc  # запоминаем старое значение количества компонент
        # проверка условий для каждого элемента
        for j in range(self.Nc):
            if (stdmax[j] > self.gs) and (((self.D[j] > self.D_common) and (len(self.S[j]) > 2 * (self.gn + 1)))
                                          or (self.Nc <= self.K / 2)):
                # расщепление центра кластера на 2
                zpol_j = [self.z[j][stdmax_inds[j]] + 0.5 * stdmax[j], *np.delete(self.z[j], [stdmax_inds[j]])]
                zotr_j = [self.z[j][stdmax_inds[j]] - 0.5 * stdmax[j], *np.delete(self.z[j], [stdmax_inds[j]])]
                self.z.remove(self.z[j])
                self.z.append(zpol_j)
                self.z.append(zotr_j)
                self.Nc = len(self.z)  # обновляем длину массива центров
        if self.Nc == Nc_old:
            self.step11(iter2)  # если расщепление не произошло, то переходим к 11 шагу
        else:
            self.step2(iter2)  # если произошло, то ко 2 шагу

    def step11(self, iter):
        iter3 = iter
        z_and_dist = []  # массив содержащий пары: координаты 2х точек и вычисленное расстояние между ними
        for i in range(self.Nc - 1):
            for j in range(i + 1, self.Nc):
                para = [(self.z[i], self.z[j]), np.linalg.norm(self.z[i] - self.z[j],
                                                               axis=-1)]  # массив содержащий 2 точки и расстояние между ними
                z_and_dist.append(para)

        # Шаг 12
        # массив содержащий пары точек и расстояния между ними, осортированный по расстояниям. выборка с расстояниями меньшими gc
        sort_z_and_dist = [row for row in sorted(z_and_dist, key=lambda x: x[1]) if row[1] < self.gc]
        z_sl = [sort_z_and_dist[j][0] for j in range(len(sort_z_and_dist))]  # массив пар точек которые будут сливаться
        # максимальное число пар точек, которые можно объединить
        inds_z_sl = []  # массив, попарно содержащий индексы точек для слияния относительно их положения в массиве z
        for p in z_sl:
            indx = [np.argwhere((self.z == p[i]).all(axis=1))[0][0] for i in range(len(p))]
            inds_z_sl.append(indx)

        # Шаг 13
        used_points = []
        if len(z_sl) != 0:
            for m in range(len(z_sl)):
                if m < self.L:
                    if all((z_sl[l][0] not in used_points) and (z_sl[l][1] not in used_points) for l in
                           range(len(z_sl))):  # учет неповторов
                        z_new_l = (z_sl[m][0] * len(self.S[inds_z_sl[m][0]]) + z_sl[m][1] * len(
                            self.S[inds_z_sl[m][1]])) / (len(self.S[inds_z_sl[m][0]]) + len(self.S[inds_z_sl[m][1]]))
                        used_points.append(z_sl[m][0])
                        used_points.append(z_sl[m][1])
                        deleted_z1 = self.z[inds_z_sl[m][0]]
                        deleted_z2 = self.z[inds_z_sl[m][1]]
                        self.z.remove(deleted_z1)
                        self.z.remove(deleted_z2)
                        self.z.append(z_new_l)
                        self.Nc = len(self.z)
        # на выходе получается self.z с новыми точками

        # Шаг 14
        if iter3 == self.I:
            print(f'центры кластеров:\n{self.z}\nкластеры:\n{self.S}')
            print('Алгоритм завершён')

        elif iter3 < self.I:
            print(f'центры кластеров:\n{self.z}\nкластеры:\n{self.S}')
            change_parametres = input('Изменить параметры?(Y/N) ')
            if change_parametres == 'Y':
                self.K = int(input('K = '))
                self.gn = int(input('gn = '))
                self.gs = int(input('gs = '))
                self.gc = int(input('gc = '))
                self.L = int(input('L = '))
            self.step2(iter3)


x0 = np.array([[0, 0], [1, 1], [2, 2], [4, 3], [5, 3], [4, 4], [5, 4], [6, 5]])
z0 = np.array([[0, 0]])

new = ISODATA(x0, z0, 2, 1, 1, 4, 0, 4)
new.start()
