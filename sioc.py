import socketio
import cv2

sio = socketio.Client()


@sio.event
def connect():
    print('connection established')
    sio.emit('message', 'HELLO!')


@sio.event
def message(data):
    print('message received with ', data)
    # sio.emit('message', {'response': 'my response'})


@sio.event
def disconnect():
    print('disconnected from server')


# camera = cv2.VideoCapture(0)
sio.connect('http://localhost:5000')
sio.wait()
