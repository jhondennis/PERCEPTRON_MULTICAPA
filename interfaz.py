from tkinter import *
import numpy as np
import multilayer_perceptron as perceptron

"""
Controles:
-Con el click del mouse se selecciona una celda.
-Presionando una vez el click derecho, se activa el modo selección,
por donde pase el puntero se seleccionara una celda o se desactivará si esta
ya estaba previamente activada.
-Presionandolo otra vez este se desactiva.
"""

##MLP  Parameters##
# Input matrix
GRID_WIDTH = 5
GRID_HEIGHT = 6
# Layer
HIDDEN_LAYER = 10
# Train settings
LEARN_RATE = 0.5
TRAIN_EPOCHS = 2500
###################
##GUI Options##
CELL_SIZE = 5
COLOR_ENABLED = '#1EFF1A'
COLOR_DISABLED = '#35A4E7'

TEXT_TITLE = "PERCEPTRON MULTICAPA"
TEXT_TRAIN = "Entrenar"
TEXT_PREDICT = "Predecir"

TEXT_FORGET = "Olvidar"
TEXT_FORGET_DESC = "Reinicia la red"

TEXT_CLEAR = "Limpiar"
TEXT_CLEAR_DESC = "Resetea los valores de la grilla"

TEXT_REFORCE = "Refuerzo"
TEXT_REFORCE_DESC = "Por cada entrenamiento, refuerza el \
aprendizaje repasando nuevamente\n"
TEXT_REFORCE_DESC += "las muestras presentadas anteriormente."


class InputGrid(Frame):

    def __init__(self, width=5, height=4):
        super().__init__()
        self.width = width
        self.height = height
        self.button_grid = {}
        self.selection_mode = False
        self.predicting_control = False
        self.init_grid()
        self.grid(row=1, column=0)

    def addButton(self, text):
        # Genera un boton y asigna evento de estado
        boton = Button(self,
                       width=CELL_SIZE,
                       height=int(CELL_SIZE/2),
                       bg=COLOR_DISABLED,
                       bd=5,
                       # text=str(text)
                       )
        boton.bind('<Enter>', lambda x: self.update_input(boton, mode='enter'))
        boton.bind('<ButtonPress-3>',
                   lambda x: self.update_input(boton, mode='m_selection'))
        boton.bind('<ButtonPress-1>',
                   lambda x: self.update_input(boton, mode='normal'))

        return boton

    def draw(self, inputs):
        # Dibuja patrón proveniente de la red
        for button, input_ in zip(self.button_grid, inputs):
            self.update_input(button, value=1 if input_ >= 0.9 else 0)
        self.predicting_control = True

    def update_input(self, boton, value=None, mode=None):
        # Estado de botón al hacerle click, 0 o 1
        if(mode == 'enter' and self.selection_mode is not True):
            return
        if(mode == 'm_selection'):
            if not self.selection_mode:
                self.selection_mode = True
            else:
                self.selection_mode = False
                return
        if(mode == 'normal'):
            self.selection_mode = False
        if(self.predicting_control is True):
            # resetea la grilla luego de predecir para
            # no tener que andar olvidando cada vez
            self.predicting_control = False
            self.reset()

        if(value is not None):
            boton['bg'] = COLOR_ENABLED if value == 1 else COLOR_DISABLED
            self.button_grid[boton] = value

        elif(self.button_grid[boton] == 1):
            boton['bg'] = COLOR_DISABLED
            self.button_grid[boton] = 0
        else:
            self.button_grid[boton] = 1
            boton['bg'] = COLOR_ENABLED
        boton.update()

    def init_grid(self):
        # Inicializa la grilla
        control = 0
        for x in range(self.width):
            for y in range(self.height):
                button = self.addButton(control)
                button.grid(row=y, column=x)
                self.button_grid[button] = 0
                control += 1

    def reset(self):
        # Resetea todos los botones de la grilla
        for button in self.button_grid:
            self.button_grid[button] = 0
            button['bg'] = COLOR_DISABLED

    def get_inputs(self):
        # Obtiene la matriz de la grilla
        return [w for w in self.button_grid.values()]


class CreateToolTip(object):
    '''
    create a tooltip for a given widget
    '''

    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.close)

    def enter(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(self.tw, text=self.text, justify='left',
                      background='white', relief='solid', borderwidth=1,
                      font=("times", "10", "normal"))
        label.pack(ipadx=1)

    def close(self, event=None):
        if self.tw:
            self.tw.destroy()


class App(object):
    def __init__(self):
        self.window = Tk()
        self.panel_inputs = InputGrid(GRID_WIDTH, GRID_HEIGHT)
        self.Perceptron = None
        self.reinforcement_data = IntVar()
        self.input_data = []
        self.output_data = []
        self.init_panel()
        self.init_perceptron()
        self.init_window()

    def init_window(self):
        screen_width = self.window.winfo_screenwidth()//2
        screen_height = self.window.winfo_screenheight()//2
        x = int((screen_width - self.window.winfo_reqwidth()) / 2)
        y = int((screen_height - self.window.winfo_reqheight()) / 2)
        self.window.withdraw()
        self.window.update_idletasks()
        self.window.geometry(f"+{x}+{y}")
        self.window.title(TEXT_TITLE)
        self.window.resizable(False, False)
        self.window.deiconify()
        self.window.mainloop()

    def predict(self):
        result = self.Perceptron.input(self.panel_inputs.get_inputs())
        self.clear()
        print("Salida actual de la red:")
        print(np.around(result, decimals=GRID_WIDTH*GRID_HEIGHT))
        self.panel_inputs.draw(result)

    def train(self):
        print("Aprendiendo...", end="")
        panel_inputs = self.panel_inputs.get_inputs()
        net_input = np.array(panel_inputs)
        net_expect = np.array(panel_inputs)
        self.input_data.append(net_input)
        self.output_data.append(net_expect)
        if not self.reinforcement_data.get():
            self.Perceptron.train(np.asarray([net_input]),
                                  np.asarray([net_expect]))
        else:
            self.Perceptron.train(np.asarray(self.input_data),
                                  np.asarray(self.output_data),
                                  epoch_number=TRAIN_EPOCHS)
        print(f"Minimo local conseguido:", self.Perceptron.errors[-1])
        print(" Listo!")
        self.panel_inputs.reset()

    def init_perceptron(self):
        input_len = len(self.panel_inputs.get_inputs())
        # Red neuronal de 3 capas
        # 1: capa oculta de HIDDEN_LAYER neuronas,
        #   con (GRID_WEIGHT*GRID_HEIGHT) entradas cada una
        # 2: salida de (GRID_WEIGHT*GRID_HEIGHT) neuronas
        model = [
            perceptron.NeuronLayer(input_len, HIDDEN_LAYER),
            perceptron.NeuronLayer(HIDDEN_LAYER, input_len),
        ]
        print(f"Red Neuronal inicializada.")
        self.Perceptron = perceptron.NeuralNetwork(model)
        self.Perceptron.learn_rate = LEARN_RATE
        self.panel_inputs.reset()

    def forget(self):
        print("Olvidando...")
        self.init_perceptron()

    def clear(self):
        self.panel_inputs.reset()

    def init_panel(self):
        control = Frame(self.window)
        Button(control, text=TEXT_TRAIN,
               command=lambda: self.train()).pack(side=LEFT)
        Button(control, text=TEXT_PREDICT,
               command=lambda: self.predict()).pack(side=LEFT)

        clear_btn = Button(control, text=TEXT_CLEAR,
                           command=lambda: self.clear())
        CreateToolTip(clear_btn, TEXT_CLEAR_DESC)
        clear_btn.pack(side=LEFT)

        forget_btn = Button(control, text=TEXT_FORGET,
                            command=lambda: self.forget())
        CreateToolTip(forget_btn, TEXT_FORGET_DESC)
        forget_btn.pack(side=LEFT)

        reinforcement_chk = Checkbutton(
            control, text=TEXT_REFORCE, variable=self.reinforcement_data)
        reinforcement_chk.select()
        CreateToolTip(reinforcement_chk, TEXT_REFORCE_DESC)
        reinforcement_chk.pack(side=LEFT)

        control.grid(column=0, row=0)


if __name__ == '__main__':
    app = App()
