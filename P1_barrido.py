# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:36:24 2018

@author: Marco
"""



#%%
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import threading
import numpy.fft as fft
import datetime
import time
import matplotlib.pylab as pylab
from scipy import signal
import os
from sys import stdout

    
    
params = {'legend.fontsize': 'medium',
     #     'figure.figsize': (15, 5),
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)


def barra_progreso(paso,pasos_totales,leyenda,tiempo_ini):

    """
    Esta función genera una barra de progreso
    
    Parametros:
    -----------
    paso : int, paso actual
    pasos_totales : int, cantidad de pasos totales
    leyenda : string, leyenda de la barra
    tiempo_ini : datetime, tiempo inicial de la barra en formato datetime   
    
    Autores: Leslie Cusato, Marco Petriella
    """    
    n_caracteres = 30    
    barrita = int(n_caracteres*(paso+1)/pasos_totales)*chr(9632)    
    ahora = datetime.datetime.now()
    eta = (ahora-tiempo_ini).total_seconds()
    eta = (pasos_totales - paso - 1)/(paso+1)*eta
       
    stdout.write("\r %s %s %3d %s |%s| %s %6.1f %s" % (leyenda,':',  int(100*(paso+1)/pasos_totales), '%', barrita.ljust(n_caracteres),'ETA:',eta,'seg'))
    
    if paso == pasos_totales-1:
        print("\n")    
    
    
    

def function_generator(parametros):
    
    """
    Esta función genera señales de tipo seno, cuadrada y rampa.
    
    Parametros:
    -----------
    Para el ingreso de los parametros de adquisición se utiliza un diccionario.

    fs : int, frecuencia de sampleo de la placa de audio. Valor máximo 44100*8 Hz. [Hz] 
    frec : float, frecuencia de la señal. [Hz] 
    amplitud : float, amplitud de la señal.
    duracion : float, tiempo de duración de la señal. [seg]
    tipo : {'square', 'sin', 'ramp', 'constant'}, tipo de señal.   
    
    Salida (returns):
    -----------------
    output_signal : numpy array, señal de salida.
    
    Autores: Leslie Cusato, Marco Petriella
    """

    fs = parametros['fs']
    frec = parametros['frec']
    amplitud = parametros['amplitud']
    duracion = parametros['duracion']
    tipo = parametros['tipo']
        
    if tipo is 'sin':
        output_signal = (amplitud*np.sin(2*np.pi*np.arange(int(duracion*fs))*frec/fs)).astype(np.float32)  
    elif tipo is 'square':
        output_signal = amplitud*signal.square(2*np.pi*np.arange(int(duracion*fs))*frec/fs, duty=0.5).astype(np.float32) 
    elif tipo is 'ramp':
        output_signal = amplitud*signal.sawtooth(2*np.pi*np.arange(int(duracion*fs))*frec/fs, width=0.5).astype(np.float32) 
    elif tipo is 'constant':
        output_signal = amplitud*(int(duracion*fs)).astype(np.float32) 

    return output_signal


def play_rec(parametros):
    
    
    """
    Descripción:
    ------------
    Esta función permite utilizar la placa de audio de la pc como un generador de funciones / osciloscopio
    con dos canales de entrada y dos de salida. Para ello utiliza la libreria pyaudio y las opciones de write() y read()
    para escribir y leer los buffer de escritura y lectura. Para realizar el envio y adquisición simultánea de señales, utiliza
    un esquema de tipo productor-consumidor que se ejecutan en thread o hilos diferenntes. Para realizar la comunicación 
    entre threads y evitar overrun o sobreescritura de los datos del buffer de lectura se utilizan dos variables de tipo block.
    El block1 se activa desde proceso productor y avisa al consumidor que el envio de la señal ha comenzado y que por lo tanto 
    puede iniciar la adquisición. 
    El block2 se activa desde el proceso consumidor y aviso al productor que la lesctura de los datos ha finalizado y por lo tanto
    puede comenzar un nuevo paso del barrido. 
    Teniendo en cuenta que existe un retardo entre la señal enviada y adquirida, y que existe variabilidad en el retardo; se puede
    utilizar el canal 0 de entrada y salida para el envio y adquisicón de una señal de disparo que permita sincronizar las mediciones.
    Notar que cuando se pone output_channel = 1, en la segunda salida pone la misma señal que en el channel 1 de salida.
    
    Parámetros:
    -----------
    Para el ingreso de los parametros de adquisición se utiliza un diccionario.
    
    parametros = {}
    parametros['fs'] : int, frecuencia de sampleo de la placa de audio. Valor máximo 44100*8 Hz. [Hz]
    parametros['input_channels'] : int, cantidad de canales de entrada.
    parametros['data_send'] : numpy array dtype=np.float32, array de tres dimensiones de la señal a enviar [cantidad_de_pasos][muestras_por_paso][output_channels]
    parametros['corrige_retardos'] = {'si','no'}, corrige el retardo utilizando la función sincroniza_con_trigger
    
    Salida (returns):
    -----------------
    data_acq: numpy array, array de tamaño [cantidad_de_pasos][muestras_por_pasos_input][input_channels]
    retardos: numpy_array, array de tamaño [cantidad_de_pasos] con los retardos entre trigger enviado y adquirido
    
    Las muestras_por_pasos está determinada por los tiempos de duración de la señal enviada y adquirida. El tiempo entre 
    muestras es 1/fs.
    
    Ejemplo:
    --------
    
    parametros = {}
    parametros['fs'] = 44100
    parametros['input_channels'] = 2
    parametros['data_send'] = 2   
    parametros['corrige_retardos'] = 'si'
    
    data_acq, retardos = play_rec(parametros)
   
    Autores: Leslie Cusato, Marco Petriella    
    """    
    
    fs = parametros['fs']
    data_send = parametros['data_send']
    input_channels = parametros['input_channels']
    corrige_retardos = parametros['corrige_retardos']
    steps = data_send.shape[0]
    
    # Cargo parametros comunes a los dos canales  
    duration_sec_send = data_send.shape[1]/fs
    output_channels = data_send.shape[2]
            
    # Obligo a la duracion de la adquisicion > a la de salida    
    duration_sec_acq = duration_sec_send + 0.2 
    
    # Inicia pyaudio
    p = pyaudio.PyAudio()
    
    # Defino los buffers de lectura y escritura
    chunk_send = int(fs*duration_sec_send)
    chunk_acq = int(fs*duration_sec_acq)
    
    # Donde se guardan los resultados                     
    data_acq = np.zeros([data_send.shape[0],chunk_acq,input_channels],dtype=np.int32)      
    
    # Defino el stream del parlante
    stream_output = p.open(format=pyaudio.paFloat32,
                    channels = output_channels,
                    rate = fs,
                    output = True,                  
    )
    
    # Defino un buffer de lectura efectivo que tiene en cuenta el delay de la medición
    chunk_delay = int(fs*stream_output.get_output_latency()) 
    chunk_acq_eff = chunk_acq + chunk_delay
    
    # Defino el stream del microfono
    stream_input = p.open(format = pyaudio.paInt32,
                    channels = input_channels,
                    rate = fs,
                    input = True,
                    frames_per_buffer = chunk_acq_eff*p.get_sample_size(pyaudio.paInt32),
    )
    
    # Defino los semaforos para sincronizar la señal y la adquisicion
    lock1 = threading.Lock() # Este lock es para asegurar que la adquisicion este siempre dentro de la señal enviada
    lock2 = threading.Lock() # Este lock es para asegurar que no se envie una nueva señal antes de haber adquirido y guardado la anterior
    lock1.acquire() # Inicializa el lock, lo pone en cero.
       
    # Defino el thread que envia la señal          
    def producer(steps):  
        for i in range(steps):
                                      
            # Genero las señales de salida para los canales
            samples = np.zeros([output_channels,4*chunk_send],dtype = np.float32)
            for j in range(output_channels):
                    samples[j,0:chunk_send] = data_send[i,:,j]
              
            # Paso la salida a un array de una dimension
            samples_out = np.reshape(samples,4*chunk_send*output_channels,order='F')
                    
            # Se entera que se guardó el paso anterior (lock2), avisa que comienza el nuevo (lock1), y envia la señal
            lock2.acquire() 
            lock1.release() 
            stream_output.start_stream()
            stream_output.write(samples_out)
            stream_output.stop_stream()        
    
        producer_exit[0] = True  
            

    # Defino el thread que adquiere la señal   
    def consumer(steps):
        tiempo_ini = datetime.datetime.now()
        for i in range(steps):           
            
            # Toma el lock, adquiere la señal y la guarda en el array
            lock1.acquire()
            stream_input.start_stream()
            stream_input.read(chunk_delay)  
            data_i = stream_input.read(chunk_acq)  
            stream_input.stop_stream()   
                
            data_i = -np.frombuffer(data_i, dtype=np.int32)                            
                
            # Guarda la salida                   
            for j in range(input_channels):
                data_acq[i,:,j] = data_i[j::input_channels]                   
                    
            # Barra de progreso
            barra_progreso(i,steps,'Progreso barrido',tiempo_ini)          
                                
            lock2.release() # Avisa al productor que terminó de escribir los datos y puede comenzar con el próximo step
    
        consumer_exit[0] = True  
           
    # Variables de salida de los threads
    producer_exit = [False]   
    consumer_exit = [False] 
            
    # Inicio los threads    
    t1 = threading.Thread(target=producer, args=[steps])
    t2 = threading.Thread(target=consumer, args=[steps])
    t1.start()
    t2.start()
             
    while(not producer_exit[0] or not consumer_exit[0]):
        time.sleep(0.2)
         
    stream_input.close()
    stream_output.close()
    p.terminate()   
    
    retardos = np.array([])
    if corrige_retardos is 'si':
        parametros_retardo = {}
        parametros_retardo['data_send'] = data_send
        parametros_retardo['data_acq']  = data_acq        
        data_acq, retardos = sincroniza_con_trigger(parametros_retardo)       
    
    return data_acq, retardos
 


def sincroniza_con_trigger(parametros):
    
    """
    Esta función corrige el retardo de las mediciones adquiridas con la función play_rec. Para ello utiliza la señal de 
    trigger enviada y adquirida en el canal 0 de la placa de audio, y sincroniza las mediciones de todos los canales de entrada. 
    El retardo se determina a partir de realizar la correlación cruzada entre la señal enviada y adquirida, y encontrando la posición
    del máximo del resultado.
    
    
    Parámetros:
    -----------
    data_acq: numpy array, array de tamaño [cantidad_de_pasos][muestras_por_pasos_input][input_channels]
    data_send: numpy array, array de tamaño [cantidad_de_pasos][muestras_por_pasos_output][output_channels]
    
    Salida (returns):
    -----------------
    data_acq_corrected : numpy array, señal de salida con retardo corregido de tamaño [cantidad_de_pasos][muestras_por_pasos_input][input_channels]. 
                         El tamaño de la segunda dimensión es la misma que la de data_send.
    retardos : numpy array, array con los retardos de tamaño [cantidad_de_pasos].
    
    Autores: Leslie Cusato, Marco Petriella   
    """
    
    data_send = parametros['data_send']
    data_acq = parametros['data_acq']   
    
    extra = 0
        
    data_acq_corrected = np.zeros([data_send.shape[0],data_send.shape[1]+extra,data_acq.shape[2]])
    retardos = np.array([])
 
    trigger_send = data_send[:,:,0]
    trigger_acq = data_acq[:,:,0]           
    
    tiempo_ini = datetime.datetime.now()
             
    for i in range(data_acq.shape[0]):
        barra_progreso(i,data_acq.shape[0],u'Progreso corrección',tiempo_ini) 

        corr = np.correlate(trigger_send[i,:] - np.mean(trigger_send[i,:]),trigger_acq[i,:] - np.mean(trigger_acq[i,:]),mode='full')
        pos_max = trigger_acq.shape[1] - np.argmax(corr)-1
        retardos = np.append(retardos,pos_max)
                    
        for j in range(data_acq.shape[2]):
            data_acq_corrected[i,:,j] = data_acq[i,pos_max:pos_max+trigger_send.shape[1]+extra,j]
        
        
    return data_acq_corrected, retardos

#%%
    


# Genero matriz de señales: ejemplo de barrido en frecuencias en el canal 0
fs = 44100 *8  
duracion = 0.2
muestras = int(fs*duracion)
canales = 2
amplitud = 0.3
frec_ini = 500
frec_fin = 25000
pasos = 40
delta_frec = (frec_fin-frec_ini)/(pasos+1)
data_send = np.zeros([pasos,muestras,canales])

for i in range(pasos):
    parametros_signal = {}
    parametros_signal['fs'] = fs
    parametros_signal['amplitud'] = amplitud
    parametros_signal['frec'] = frec_ini + i*delta_frec
    parametros_signal['duracion'] = duracion
    parametros_signal['tipo'] = 'sin'
    
    output_signal = function_generator(parametros_signal)
    output_signal = output_signal*np.arange(muestras)/muestras
    data_send[i,:,0] = output_signal


# Realiza medicion
parametros = {}
parametros['fs'] = fs
parametros['input_channels'] = 2
parametros['corrige_retardos'] = 'si'
parametros['data_send'] = data_send

data_acq, retardos = play_rec(parametros)


plt.plot(np.transpose(data_acq[0,:,0]))


#%%

### Corrige retardo y grafica
#parametros_retardo = {}
#parametros_retardo['data_send'] = data_send
#parametros_retardo['data_acq']  = data_acq
#
#data_acq, retardos = sincroniza_con_trigger(parametros_retardo)

#%%
ch = 0
step = 10

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .12, .75, .8])
ax1 = ax.twinx()
ax.plot(np.transpose(data_acq[step,:,ch]),'-',color='r', label='señal adquirida',alpha=0.5)
ax1.plot(np.transpose(data_send[step,:,ch]),'-',color='b', label='señal enviada',alpha=0.5)
ax.legend(loc=1)
ax1.legend(loc=4)
plt.show()


fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.1, .1, .75, .8])
ax.hist(retardos/fs*1000, bins=100)
ax.set_title(u'Retardos')
ax.set_xlabel('Retardos [ms]')
ax.set_ylabel('Frecuencia')


#%%
### ANALISIS de la señal adquirida. Cheque que la señal adquirida corresponde a la enviada

fs = parametros['fs']

ch_acq = 0
ch_send = 0
paso = 5

### Realiza la FFT de la señal enviada y adquirida
fft_send = abs(fft.fft(data_send[paso,:,ch_send]))/int(data_send.shape[1]/2+1)
fft_send = fft_send[0:int(data_send.shape[1]/2+1)]
fft_acq = abs(fft.fft(data_acq[paso,:,ch_acq]))/int(data_acq.shape[1]/2+1)
fft_acq = fft_acq[0:int(data_acq.shape[1]/2+1)]

frec_send = np.linspace(0,int(data_send.shape[1]/2),int(data_send.shape[1]/2+1))
frec_send = frec_send*(fs/2+1)/int(data_send.shape[1]/2+1)
frec_acq = np.linspace(0,int(data_acq.shape[1]/2),int(data_acq.shape[1]/2+1))
frec_acq = frec_acq*(fs/2+1)/int(data_acq.shape[1]/2+1)

fig = plt.figure(figsize=(14, 7), dpi=250)
ax = fig.add_axes([.12, .12, .75, .8])
ax1 = ax.twinx()
ax.plot(frec_send,fft_send,'-' ,label='Frec enviada',alpha=0.7)
ax1.plot(frec_acq,fft_acq,'-',color='red', label=u'Señal adquirida',alpha=0.7)
ax.set_title(u'FFT de la señal enviada y adquirida')
ax.set_xlabel('Frecuencia [Hz]')
ax.set_ylabel('Amplitud [a.u.]')
ax.legend(loc=1)
ax1.legend(loc=4)
plt.show()

 

#%%

Is = 1.0*1e-12
Vt = 26.0*1e-3
n = 1.

Vd = np.linspace(-1,1,1000)
Id = Is*(np.exp(Vd/n/Vt)-1)

Rs = 100
Vs = 1
Ir = Vs/Rs - Vd/Rs


plt.plot(Vd,Id)
plt.plot(Vd,Ir)

    
