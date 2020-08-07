import PROJECT
from keras.models import load_model
from collections import deque

inpuut = r"F:\Data\ActivityNet\Dodgeball\dodgeball.mp4"
output = r"F:\Project"

moodel = load_model(moddel)
lb = pickle.loads(open("label","rb").read())

mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=128)



vs = cv2.VideoCapture(inpuut)

(W,H) = (None,None)

while True:
    (grabbed,frame) = vs.read()

    if not grabbed:
        break

    if W is None or H is None:
        (H,W) = frame.shape[:2]

    output = frame.copy()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame,(224,224).astype("float32"))
    frame-=mean

    preds = moodel.predict(np.expand_dims(frame,axis=0))[0]
    Q.append(preds)

    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = lb.classes_[i]

    text = "activity : {}".format(label)
    cv2.putText(output,text,(35,50),cv2.FONT_HERSHEY_SIMPLEX,
                1.25,(0,255,0),5)

    cv2.imshow(output)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

vs.release()
