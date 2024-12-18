import pygame
from queue import PriorityQueue
import math

# Configuraciones iniciales
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("A*")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
AMARILLO = (255, 255, 0)


class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []

    def get_pos(self):
        return self.fila, self.col

    def es_cerrado(self):
        return self.color == ROJO

    def es_abierto(self):
        return self.color == VERDE

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_cerrado(self):
        self.color = ROJO

    def hacer_abierto(self):
        self.color = VERDE

    def hacer_camino(self):
        self.color = AMARILLO

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

    def actualizar_vecinos(self, grid):
        self.vecinos = []

        if self.fila < self.total_filas - 1 and not grid[self.fila + 1][self.col].es_pared():  # Abajo
            self.vecinos.append(grid[self.fila + 1][self.col])

        if self.fila > 0 and not grid[self.fila - 1][self.col].es_pared():  # Arriba
            self.vecinos.append(grid[self.fila - 1][self.col])

        if self.col < self.total_filas - 1 and not grid[self.fila][self.col + 1].es_pared():  # Derecha
            self.vecinos.append(grid[self.fila][self.col + 1])

        if self.col > 0 and not grid[self.fila][self.col - 1].es_pared():  # Izquierda
            self.vecinos.append(grid[self.fila][self.col - 1])

        # Movimientos diagonales
        if self.fila > 0 and self.col > 0 and not grid[self.fila - 1][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col - 1])  # Arriba izquierda

        if self.fila > 0 and self.col < self.total_filas - 1 and not grid[self.fila - 1][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col + 1])  # Arriba derecha

        if self.fila < self.total_filas - 1 and self.col > 0 and not grid[self.fila + 1][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col - 1])  # Abajo izquierda

        if self.fila < self.total_filas - 1 and self.col < self.total_filas - 1 and not grid[self.fila + 1][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col + 1])  # Abajo derecha


def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid


def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))


def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()


def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col


def new_way(padre, nodo_actual, dibujar):
    while nodo_actual in padre:
        nodo_actual = padre[nodo_actual]
        nodo_actual.hacer_camino()
        dibujar()

def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def calcular_costo(padre, nodo_actual):
    x1, y1 = padre.get_pos()
    x2, y2 = nodo_actual.get_pos()
    if abs(x1 - x2) == 1 and abs(y1 - y2) == 1:  # Movimiento diagonal
        return math.sqrt(2)
    return 1  # Movimiento horizontal o vertical

def a_asterisco(dibujar, grid, inicio, fin):
    contador = 0
    open_set = PriorityQueue()
    open_set.put((0, contador, inicio))
    padre = {}
    g_acomu = {nodo: float("inf") for fila in grid for nodo in fila}
    g_acomu[inicio] = 0
    f_acom = {nodo: float("inf") for fila in grid for nodo in fila}
    f_acom[inicio] = h(inicio.get_pos(), fin.get_pos())

    open_set_hash = {inicio}
    closed_list = []

    while not open_set.empty():
        pygame.time.delay(50)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        nodo_actual = open_set.get()[2]
        open_set_hash.remove(nodo_actual)

        if nodo_actual == fin:
            new_way(padre, fin, dibujar)
            fin.hacer_fin()
            print("Lista cerrada:", [n.get_pos() for n in closed_list])
            return True

        for vecino in nodo_actual.vecinos:
            temp_g_acomu = g_acomu[nodo_actual] + calcular_costo(nodo_actual, vecino)

            if temp_g_acomu < g_acomu[vecino]:
                padre[vecino] = nodo_actual
                g_acomu[vecino] = temp_g_acomu
                f_acom[vecino] = temp_g_acomu + h(vecino.get_pos(), fin.get_pos())
                if vecino not in open_set_hash:
                    contador += 1
                    open_set.put((f_acom[vecino], contador, vecino))
                    open_set_hash.add(vecino)
                    vecino.hacer_abierto()

        dibujar()

        if nodo_actual != inicio:
            nodo_actual.hacer_cerrado()
            closed_list.append(nodo_actual)

    print("No se encontró un camino. Lista cerrada:", [n.get_pos() for n in closed_list])
    return False


def main(ventana, ancho):
    FILAS = 9
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None

    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:  # Clic izquierdo
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()
                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()
                elif nodo != inicio and nodo != fin:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Clic derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)

                    a_asterisco(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)

                if event.key == pygame.K_c:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)

    pygame.quit()


main(VENTANA, ANCHO_VENTANA)
