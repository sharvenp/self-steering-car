"""
Author: Sharven

TODO:

- Car Mechanics (Done)
- Track Generation (Done)
- Ray Casting for car vision (Done)
- RL Agent
"""

from PIL import Image, ImageDraw
import pygame as pg
import math as m
import time
import random as r
import time as t
import numpy as np

from agent import Agent

rad = m.radians
deg = m.degrees

def get_sign(x):
    if x != 0: 
        return abs(x)//x
    return 0

def rad_to_deg(x):
    return (180/m.pi) * x

class Road(pg.sprite.Sprite):

    def __init__(self):
        super().__init__()

        self.image = pg.image.load('resources/road.png').convert_alpha()
        self.wall_mask = pg.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        pic = Image.open('resources/road.png').convert('RGB')
        self.binary_map = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 3)[:,:,1]//150

class Car(pg.sprite.Sprite):

    def __init__(self, x, y):

        super().__init__()

        self.image = pg.image.load('resources/car.png').convert_alpha()
        self.clean_image = self.image        
        self.car_mask = pg.mask.from_surface(self.image)
        self.rect = self.image.get_rect()

        self.x = x
        self.y = y

        self.MAX_VELOCITY = 5
        self.ANGULAR_VELOCITY = 7.5
        self.ACCELERATION = 4
        self.DRAG = 0.2
        self.RAY_LENGTH = 150
        self.RAY_ANGLE = 90
        self.N_RAY = 4
        self.raycast_step = (self.RAY_ANGLE/(self.N_RAY-1))

        self.curr_dx = 0
        self.curr_dy = 0
        self.angle = 0

    def rotate(self, image, rect, rotation):

        # Rotate the image around a center
        rot_img = pg.transform.rotate(image, -rotation)
        rect = rot_img.get_rect(center=rect.center)
        return rot_img, rect

    def drive(self, inp):

        engine = inp[0]
        turn = inp[1]

        # Rotation Calculation
        rot = turn * self.ANGULAR_VELOCITY * -1
        if self.curr_dx or self.curr_dy:
            self.angle += rot

        if self.angle > 360:
            self.angle = 0
        elif self.angle < 0:
            self.angle = 360

        self.image, self.rect = self.rotate(self.clean_image, self.rect, self.angle)

        # Movement Calculation
        curr_x, curr_y = self.rect[0], self.rect[1]

        dx = m.cos(rad(self.angle)) * self.ACCELERATION * engine
        dy = m.sin(rad(self.angle)) * self.ACCELERATION * engine
    
        self.curr_dx += dx        
        self.curr_dy += dy

        # Add Drag
        if self.curr_dx > 0:
            self.curr_dx -= self.DRAG * self.curr_dx
            self.curr_dx = m.floor(self.curr_dx)
        elif self.curr_dx < 0:
            self.curr_dx += self.DRAG * abs(self.curr_dx)
            self.curr_dx = m.ceil(self.curr_dx)

        if self.curr_dy > 0:
            self.curr_dy -= self.DRAG * self.curr_dy
            self.curr_dy = m.floor(self.curr_dy)
        elif self.curr_dy < 0:
            self.curr_dy += self.DRAG * abs(self.curr_dy)
            self.curr_dy = m.ceil(self.curr_dy)

        # Bound Velocity
        if abs(self.curr_dx) > self.MAX_VELOCITY:
            curr_dx = get_sign(self.curr_dx)*self.MAX_VELOCITY
    
        if abs(self.curr_dy) > self.MAX_VELOCITY:
            curr_dy = get_sign(self.curr_dy)*self.MAX_VELOCITY

        # Update Position
        self.x += self.curr_dx
        self.y += self.curr_dy
        self.rect.center = (self.x, self.y)
        self.car_mask = pg.mask.from_surface(self.image)
        
        # print(self.curr_dx, self.curr_dy, self.angle)

    def _get_distance(self, x0, y0, x1, y1):

        return m.sqrt(((x1 - x0)**2) + ((y1 - y0)**2))

    def _get_blocks(self, x0, y0, x1, y1):
        
        # Bresenham's line algorithm
        blocks = []

        s = abs(x1 - x0) < abs(y1 - y0)
        if s:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = x1 - x0
        dy = abs(y1 - y0)
        e = 0
        ystep = 0
        if y1 != y0:
            ystep = (y1 - y0)//abs(y1 - y0)
        y = y0
        
        for x in range(x0, x1): 
            if s: 
                blocks.append((y, x));
            else:
                blocks.append((x, y));

            e += dy;
            if dx <= 2*e:
                y += ystep
                e -= dx

        return blocks

    def cast_ray(self, x, y, angle, bin_map):
    
        # Append all blocks that are not on a wall
        dx = m.cos(rad(angle)) * self.RAY_LENGTH
        dy = m.sin(rad(angle)) * self.RAY_LENGTH

        blocks = self._get_blocks(x, y, int(x + dx), int(y + dy))

        new_blocks = []

        if blocks[0] != (x, y):
            blocks = blocks[::-1]

        for block in blocks:

            c, r = block

            if c >= bin_map.shape[1] or r >= bin_map.shape[0] or r < 0 or c < 0:
                break
            
            if self._get_distance(x, y, c, r) > self.RAY_LENGTH:
                break

            if bin_map[r][c] != 0 and (x, y) != (c, r):
                new_blocks.append(block)
                break

            new_blocks.append(block)
            
        return new_blocks

class CarSimulation:

    def __init__(self, window_width, window_height):
        
        self.WIDTH = window_width
        self.HEIGHT = window_height

        self.BACKGROUND_COLOR = (0, 0, 0)
        self.CAR_COLOR = (255, 0, 0)
        self.TRACK_COLOR = (120, 120, 120)
        self.GRASS_COLOR = (0, 150, 0)
        self.RAY_CAST_COLOR = (0, 0, 255)

        self.FRAME_RATE = 30

        self.agent = Agent([(4, ''), (3, 'relu'), (3, 'relu'), (2, 'softmax')], True, 0.0007, 0.99, 100)
        self.LOSE_REWARD = -window_width*1.5
        self.WIN_REWARD = window_width*2

    def _line_function(self, x, m, b):
        
        # y = mx + b
        return int((m*x) + b)

    def generate_track(self, start, end, n, weave_factor):
        
        points = [start]

        # Calculate hypotenuse and delta based on n
        x = abs(end[0] - start[0])
        y = abs(end[1] - start[1])
        c = m.sqrt((x**2) + (y**2))
        sep = c/n
        dx, dy = sep, 0

        # Create a diagonal line with n points
        for i in range(1, n):
            new_point = ((points[i-1][0] + dx), (points[i-1][1] + dy))
            points.append(new_point)

        # Weave lines
        for i in range(1, len(points)):
            p = points[i]
            points[i] = (int(p[0]), int(p[1] + ((((2*r.random()) - 1) * weave_factor))))
        
        points.append(end)
        return points

    def create_track_sprite(self, filename, points, road_width):
        
        image = Image.new('RGBA', (self.WIDTH, self.HEIGHT))
        draw = ImageDraw.Draw(image)

        draw.rectangle([0, 0, self.WIDTH, self.HEIGHT], fill=self.GRASS_COLOR)

        circle_radius = road_width//2

        # Draw Track and save as sprite
        for i in range(1, len(points)):
            
            p1 = points[i - 1]
            p2 = points[i]
            dx = int(p2[0] - p1[0])
            dy = int(p2[1] - p1[1])
            slope = dy/dx
            b = p2[1] - (slope * p2[0])
            curr_x, curr_y = p1
            
            for k in range(dx):
            
                circle_x = curr_x + k
                circle_y = self._line_function(curr_x + k, slope, b)

                rect_sequence = [circle_x - circle_radius, circle_y - circle_radius, circle_x + circle_radius, circle_y + circle_radius]
                draw.ellipse(rect_sequence, fill = (0, 0, 0, 0))
        
        image.save(filename, 'PNG')
    
    def run(self, debug=False):
        
        # Init
        pg.init()
        
        screen = pg.display.set_mode((self.WIDTH, self.HEIGHT))    
        pg.display.set_caption('Car Simulation')
        clock = pg.time.Clock()
        
        font = pg.font.SysFont('Ubuntu Mono', 30)

        inp = [0, 0]
        score = 0

        start_time = time.time()
        episode = self.agent.current_episode
        simulation_number = 0

        while True:

            # Generate track based on params
            road_width = 100
            weave_factor = 110
            n = 6
            points = self.generate_track((0, self.HEIGHT//2), (self.WIDTH, self.HEIGHT//2), n, weave_factor)

            # Make car and face it in track direction
            car = Car(30, self.HEIGHT//2)
            
            name = "resources/road.png"
            self.create_track_sprite(name, points, road_width)

            road = Road()

            all_sprites = pg.sprite.Group()
            all_sprites.add(road)
            all_sprites.add(car)

            game_over = False
            states, actions, rewards = [], [], []

            while not game_over:
                
                screen.fill(self.BACKGROUND_COLOR)    

                # Exit Condition
                e = pg.event.poll()
                if e.type == pg.QUIT:
                    return
                keys = pg.key.get_pressed()
                if keys[pg.K_ESCAPE] or keys[pg.K_SLASH]:
                    return

                # Keyboard Input
                # inp[0] = int(keys[pg.K_w])
                # inp[1] = int(keys[pg.K_a]) - int(keys[pg.K_d])            

                # Draw Vision
                vision_data = []
                phi = (car.angle - (car.RAY_ANGLE/2)) % 360
                while phi != (car.angle + (car.RAY_ANGLE/2) + car.raycast_step) % 360:
                    blocks = car.cast_ray(int(car.x), int(car.y), phi, road.binary_map)
                    if debug:
                        for block in blocks:
                            pg.draw.rect(screen, self.RAY_CAST_COLOR, (block[0], block[1], 1, 1))
                    vision_data.append(car._get_distance(int(car.x), int(car.y), blocks[-1][0], blocks[-1][1])/car.RAY_LENGTH)

                    phi += car.raycast_step
                    phi %= 360

                states.append(vision_data)
                action = self.agent._get_state_action(vision_data)
                actions.append(action)
                inp[0] = 0.8 # constant speed
                inp[1] = (2 * action) - 1

                # Show Engine and Turn as text
                engine_text = font.render('Engine: ' + str(inp[0]), False, (255, 255, 255))
                turn_text = font.render('Turn: ' + str(inp[1]), False, (255, 255, 255))

                car.drive(inp)

                # Collision Check
                if pg.sprite.spritecollide(car, [road], False, pg.sprite.collide_mask): 
                    game_over = True
                    rewards.append(self.LOSE_REWARD)

                # Win check
                if car.x > self.WIDTH:
                    score += 1
                    game_over = True
                    rewards.append(self.WIN_REWARD)

                if not game_over:
                    rewards.append(car.x/2)

                # Debug Gizmos
                if debug:

                    # Draw track guide
                    for i in range(1, len(points)):
                        p1 = points[i - 1]
                        p2 = points[i]
                        pg.draw.line(screen, (0, 255, 0), p1, p2)

                    # Draw Collider
                    pg.draw.rect(screen, (0, 0, 255), car.rect, 2)
                
                if not game_over:
                    all_sprites.draw(screen)
                    all_sprites.update()
                    screen.blit(engine_text, (10, self.HEIGHT - 60))
                    screen.blit(turn_text, (10, self.HEIGHT - 30))
                    clock.tick(self.FRAME_RATE)
                    pg.display.flip()     

            # print("Score:", score) 
            elapsed_time = time.time() - start_time    
            time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))  
            simulation_number += 1
            episode += 1
            output_string = "Episode: {:0>5} Score: {:0>3} Reward: {:0>7} T+: {} @ {:.3f}E/s".format(episode, score, sum(rewards), time_str, simulation_number/elapsed_time)
            print(output_string)

            self.agent._train_episode(states, actions, rewards, episode)

def main():
    c = CarSimulation(700, 700)
    c.run(debug=True)

if __name__ == "__main__":
    main()
