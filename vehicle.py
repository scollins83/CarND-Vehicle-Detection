import numpy as np

class VehicleTracker():

    def __init__(self, history_length):
        self.vehicles = []
        self.history_length = history_length


    def add_box(self, box, xbox_left, ytop_draw, y_start, win_draw):
        # Determine if car is 10% or more different than any other box average in the cars list.
        # Add the new box to the first car that it matches 30% with.
        if len(self.vehicles) == 0:
            new_vehicle = Vehicle(self.history_length)
            new_vehicle.add_box(box, xbox_left, ytop_draw, y_start, win_draw)
            self.vehicles.append(new_vehicle)

        else:
            for car in self.vehicles:
                car_box = car.average_box

                if ((car_box[0][0] >= (box[0][0] - box[0][0]*.1)) and (car_box[0][0] <= (box[0][0] + box[0][0]*.1))) \
                    and \
                    ((car_box[0][1] >= (box[0][1] - box[0][1] * .1)) and (car_box[0][1] <= (box[0][1] + box[0][1] * .1))) \
                    and \
                    ((car_box[1] >= (box[1] - box[1] * .1)) and (car_box[1] <= (box[1] + box[1] * .1))) \
                    and \
                    ((car_box[2] >= (box[2] - box[2] * .1)) and (car_box[2] <= (box[2] + box[2] * .1))):
                    car.add_box(box, xbox_left, ytop_draw, y_start, win_draw)
                else:
                    new_vehicle = Vehicle(self.history_length)
                    new_vehicle.add_box(box, xbox_left, ytop_draw, y_start, win_draw)
                    self.vehicles.append(new_vehicle)


class Vehicle():

    def __init__(self, history_length):
        self.box_history = []
        self.history_length = history_length
        self.average_box = None
        self.avg_xbox_left = None
        self.avg_ytop_draw = None
        self.avg_y_start = None
        self.avg_win_draw = None
        self.xbox_left = []
        self.ytop_draw = []
        self.y_start = []
        self.win_draw = []



    def add_box(self, box, xbox_left, ytop_draw, y_start, win_draw):
        """

        :param boxes:
        :return:
        """
        print(box)
        print("Pre-box history: " + str(len(self.box_history)))

        self.box_history.append(box)
        self.xbox_left.append(xbox_left)
        self.ytop_draw.append(ytop_draw)
        self.y_start.append(y_start)
        self.win_draw.append(win_draw)

        print("Mid-box history: " + str(len(self.box_history)))

        self.adjust_box_history()

        print("Post-box history: " + str(len(self.box_history)))
        self.set_average_box()



    def adjust_box_history(self):
        """

        :return:
        """
        if len(self.box_history) > self.history_length:
            self.box_history = self.box_history[-self.history_length:]
            self.xbox_left = self.xbox_left[-self.history_length:]
            self.ytop_draw = self.ytop_draw[-self.history_length:]
            self.y_start = self.y_start[-self.history_length:]
            self.win_draw = self.win_draw[-self.history_length:]


    def set_average_box(self):
        """
        ((x, y), z)
        :return:
        """
        if len(self.box_history) > 0:

            counter = 0
            x = 0
            y = 0
            z = 0
            w = 0

            for box in self.box_history:
                counter += 1
                x += box[0][0]
                y += box[0][1]
                z += box[1]
                w += box[2]

            self.average_box = (((x//counter),(y//counter)), (z//counter), (w//counter))
            print(self.average_box)
            self.avg_xbox_left = int(np.mean(self.xbox_left))
            self.avg_ytop_draw = int(np.mean(self.ytop_draw))
            self.avg_y_start = int(np.mean(self.y_start))
            self.avg_win_draw = int(np.mean(self.win_draw))
