import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

def prompt_user(image):

    plt.imshow(image,cmap=plt.get_cmap('gray'))
    clicks = []
    clicks = np.array(plt.ginput(5,show_clicks=True), np.int32)
    np.save("clicks.npy", clicks)
    plt.close()
    #clicks=np.load("clicks.npy")
    return clicks

def interpolate_roi(clicks):
    ellipse = cv2.fitEllipse(clicks)
    (xc,yc),(d1,d2),angle = ellipse
    center = (int(xc), int(yc))
    axes = (int(d1 / 2), int(d2 / 2))
    angle = int(angle)
    points = cv2.ellipse2Poly(center, axes, angle, 0, 360, 5)
    return points

def render_roi(image, points):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    print(points.shape)
    render=cv2.drawContours(image, [points], -1, 255, 3)
    plt.imshow(render)
    plt.show()

    answer = input("OK?: Y/N")
    plt.close()
    return answer


def write_config(points):
    points_str=str(points.tolist())[1:-1] + ","

    with open("idtrackerai.conf", "r") as filehandle:
        config = json.load(filehandle)

    config["_roi"]["value"] = [[points_str]]

    with open("output.conf", "w") as filehandle:
        json.dump(config, filehandle)


def main():
    image=cv2.imread("image.png")[:,:,0]
    count=0
    
    while count < 3:
        clicks=prompt_user(image)
        # clicks=np.load("clicks.npy")
        
        points=interpolate_roi(clicks)
        answer = render_roi(image, points)

        if answer == "Y":
            break
        else:
            count+=1

    write_config(points)


if __name__ == "__main__":
    main()
