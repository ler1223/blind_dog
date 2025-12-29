import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import BasicTrain
import matplotlib.animation
from Simulation import Dog2, Dog


class MinimalVisualizer:
    """
    Минималистичная визуализация - только квадрат, точка и стрелка.
    Никаких текстов кроме информации в углу.
    """
    def __init__(self, field_size=10.0):
        self.field_size = field_size

        # Создаем фигуру
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-field_size / 2 - 1, field_size / 2 + 1)
        self.ax.set_ylim(-field_size / 2 - 1, field_size / 2 + 1)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        # Границы поля
        self.ax.add_patch(plt.Rectangle(
            (-field_size / 2, -field_size / 2),
            field_size, field_size,
            linewidth=2, edgecolor='black', facecolor='none', alpha=0.5
        ))

        # Информация
        self.info_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        # Храним только основные объекты
        self.dog_patch = None
        self.target_point = None
        self.velocity_arrow = None

    def update_scene(self, dog, target, step=0, fitness=0.0):
        """Обновляет сцену - все объекты пересоздаются заново."""
        # Очищаем все
        self.clear_scene()

        # Цель
        if hasattr(target, 'active') and target.active:
            self.target_point = self.ax.scatter(
                target.position[0], target.position[1],
                s=100, c='red', marker='o', alpha=0.7, edgecolors='darkred'
            )

        # Собака (без имени)
        dog_size = getattr(dog, 'size', 0.5)
        self.dog_patch = plt.Rectangle(
            (dog.position[0] - dog_size, dog.position[1] - dog_size),
            2 * dog_size, 2 * dog_size,
            linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.7
        )
        self.ax.add_patch(self.dog_patch)

        # Скорость (без численного текста)
        speed = np.linalg.norm(dog.velocity)
        if speed > 0.1:
            self.velocity_arrow = self.ax.arrow(
                dog.position[0], dog.position[1],
                dog.velocity[0] * 0.5, dog.velocity[1] * 0.5,
                head_width=0.2, head_length=0.3,
                fc='green', ec='darkgreen', alpha=0.6
            )

        # Информация (все данные только здесь)
        info = f"Step: {step}\n"
        info += f"Position: ({dog.position[0]:.2f}, {dog.position[1]:.2f})\n"
        info += f"Velocity: ({dog.velocity[0]:.2f}, {dog.velocity[1]:.2f})\n"
        info += f"Speed: {speed:.2f}\n"
        info += f"Fitness: {fitness:.2f}"

        if hasattr(target, 'active') and target.active:
            distance = np.linalg.norm(target.position - dog.position)
            info += f"\nDistance: {distance:.2f}"

        self.info_text.set_text(info)

        # Оси и заголовок
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Dog Simulation')

    def clear_scene(self):
        """Очищает сцену простым удалением объектов."""
        if self.dog_patch:
            self.dog_patch.remove()
            self.dog_patch = None

        if self.target_point:
            self.target_point.remove()
            self.target_point = None

        if self.velocity_arrow:
            self.velocity_arrow.remove()
            self.velocity_arrow = None

    def show(self):
        plt.tight_layout()
        plt.show()


class MinimalVisualizer2:
    """
    Минималистичная версия для быстрой визуализации.
    """

    def __init__(self, field_size=10.0):
        self.field_size = field_size
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.ax.set_xlim(-field_size / 2, field_size / 2)
        self.ax.set_ylim(-field_size / 2, field_size / 2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        # Объекты
        self.dog_rect = None
        self.target_circle = None
        self.enemy_triangles = []
        self.info_text = None

    def update(self, dog, target, enemies, step=0, reward=0):
        """Быстрое обновление."""
        # Очистка
        self.clear()

        for enemy in enemies:
            triangle = plt.Rectangle((enemy.position[0] - enemy.size / 2, enemy.position[1] - enemy.size / 2),
                                     enemy.size, enemy.size, color='orange', alpha=0.6)
            self.ax.add_patch(triangle)
            self.enemy_triangles.append(triangle)

        # Цель
        if hasattr(target, 'active') and target.active:
            self.target_circle = plt.Circle(
                (target.position[0], target.position[1]),
                target.radius, color='red', alpha=0.7
            )
            self.ax.add_patch(self.target_circle)

        # Собака
        self.dog_rect = plt.Rectangle(
            (dog.position[0] - dog.size / 2, dog.position[1] - dog.size / 2),
            dog.size, dog.size,
            color='blue', alpha=0.7
        )
        self.ax.add_patch(self.dog_rect)

        # Информация
        info = f"Step: {step}\nReward: {reward:.2f}"
        if self.info_text:
            self.info_text.remove()

        self.info_text = self.ax.text(
            0.02, 0.98, info, transform=self.ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

    def clear(self):
        """Быстрая очистка."""
        if self.dog_rect:
            self.dog_rect.remove()
        if self.target_circle:
            self.target_circle.remove()
        for triangle in self.enemy_triangles:
            triangle.remove()
        self.enemy_triangles = []

    def show(self, pause_time=0.01):
        """Быстрый показ."""
        plt.draw()
        plt.pause(pause_time)


class MinimalVisualizer3:
    """
    Минималистичная версия для быстрой визуализации.
    """

    def __init__(self, field_size=10.0):
        self.field_size = field_size
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        self.ax.set_xlim(-field_size / 2, field_size / 2)
        self.ax.set_ylim(-field_size / 2, field_size / 2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        # Объекты
        self.dog_rect = None
        self.target_circle = None
        self.enemy_triangles = []
        self.info_text = None

    def update(self, dog: Dog2, target=None, enemies=None, step=0, reward=0, feeder=None, drinking_bowl=None):
        """Быстрое обновление."""
        # Очистка
        self.clear()

        if enemies is not None:
            for enemy in enemies:
                e = plt.Rectangle((enemy.position[0] - enemy.size / 2, enemy.position[1] - enemy.size / 2),
                                         enemy.size, enemy.size, color='orange', alpha=0.6)
                self.ax.add_patch(e)
                self.enemy_triangles.append(e)

        # Цель
        if target is not None:
            if hasattr(target, 'active') and target.active:
                self.target_circle = plt.Circle(
                    (target.position[0], target.position[1]),
                    target.radius, color='red', alpha=0.7
                )
                self.ax.add_patch(self.target_circle)

        if feeder is not None:
            f = plt.Rectangle((feeder.position[0] - feeder.size / 2, feeder.position[1] - feeder.size / 2),
                              feeder.size, feeder.size, color='brown', alpha=0.6)
            self.ax.add_patch(f)

        if drinking_bowl is not None:
            d = plt.Rectangle((drinking_bowl.position[0] - drinking_bowl.size / 2, drinking_bowl.position[1] - drinking_bowl.size / 2),
                              drinking_bowl.size, drinking_bowl.size, color='blue', alpha=0.6)
            self.ax.add_patch(d)

        # Собака
        self.dog_rect = plt.Rectangle(
            (dog.position[0] - dog.size / 2, dog.position[1] - dog.size / 2),
            dog.size, dog.size,
            color='blue', alpha=0.7
        )
        self.ax.add_patch(self.dog_rect)

        # Информация
        info = f"Step: {step}\nReward: {reward:.2f}"
        if target is not None:
            info += f"\nTarget: {target.id}"
        if feeder is not None:
            info += f"\nSatiety: {dog.satiety}"
        if drinking_bowl is not None:
            info += f"\nThirst: {dog.thirst}"
        if self.info_text:
            self.info_text.remove()

        self.info_text = self.ax.text(
            0.02, 0.98, info, transform=self.ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

    def clear(self):
        """Быстрая очистка."""
        if self.dog_rect:
            self.dog_rect.remove()
        if self.target_circle:
            self.target_circle.remove()
        for triangle in self.enemy_triangles:
            triangle.remove()
        self.enemy_triangles = []

    def show(self, pause_time=0.01):
        """Быстрый показ."""
        plt.draw()
        plt.pause(pause_time)


class SimpleAnimation:
    """
    Простая анимация для симуляции в реальном времени.
    """

    def __init__(self, env, dog, individual, target, steps=200, interval=50, device="cpu"):
        """
        Инициализация анимации.

        Args:
            env: Объект среды
            dog: Объект собаки
            target: Объект цели
            steps: Количество шагов анимации
            interval: Интервал между кадрами в миллисекундах
        """
        self.env = env
        self.dog = dog
        self.individual = individual
        self.target = target
        self.steps = steps
        self.interval = interval
        self.device = device

        # Создаем визуализатор
        if env.__class__.__name__ == "Environment":
            self.visualizer = MinimalVisualizer(field_size=env.field_size)
        elif self.env.__class__.__name__ == "Environment2":
            self.visualizer = MinimalVisualizer2(field_size=env.field_size)
        elif self.env.__class__.__name__ == "Environment3":
            self.visualizer = MinimalVisualizer3(field_size=env.field_size)

        self.is_done = False
        self.current_step = 0

        # Для хранения истории
        self.history = {
            'positions': [],
            'fitness': []
        }

        # Создаем анимацию
        self.ani = FuncAnimation(
            self.visualizer.fig,
            self._update_frame,
            frames=self._frame_generator,
            interval=interval,
            repeat=False,
            blit=False,
            cache_frame_data=False
        )

    def _frame_generator(self):
        """
        Генератор кадров, который останавливается при done=True.
        """
        self.current_step = 0
        while self.current_step < self.steps and not self.is_done:
            yield self.current_step
            self.current_step += 1

    def _update_frame(self, frame):
        """
        Обновляет кадр анимации.

        Args:
            frame: Номер кадра
        """
        # Выбираем случайное ускорение для демонстрации
        with torch.no_grad():
            state = self.env.get_state(self.dog, self.target)
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            predict = self.individual.network.forward(state)
            if self.device == "cuda":
                predict = predict.cpu()
            accelerate = predict.squeeze(0)
            accelerate, angle_acceleration = accelerate[0], accelerate[1]
        self.dog, self.target, reward, done = self.env.step(self.dog, self.target, accelerate, angle_acceleration, frame, self.steps)

        # Обновляем визуализацию
        self.is_done = done

        if self.env.__class__.__name__ == "Environment":
            self.visualizer.update_scene(
                self.dog, self.target,
                step=frame,
                fitness=reward
            )
        elif self.env.__class__.__name__ == "Environment2":
            self.visualizer.update(
                self.dog, self.target,
                enemies=self.env.enemies,
                step=frame,
                reward=reward
            )
        elif self.env.__class__.__name__ == "Environment3":
            self.visualizer.update(
                self.dog, self.target,
                enemies=self.env.enemies,
                step=frame,
                reward=reward,
                feeder=self.env.feeder,
                drinking_bowl=self.env.drinking_bowl,
            )

        # Сохраняем историю
        self.history['positions'].append(self.dog.position.numpy())
        self.history['fitness'].append(reward)

        if frame == self.steps - 1 or done:
            self.ani.event_source.stop()
            plt.close(self.visualizer.fig)

    def show(self):
        """Показывает анимацию."""
        try:
            self.visualizer.show()
        except:
            pass

    def save(self, save_path, fps=30):
        """
        Сохраняет анимацию в файл.
        Args:
            save_path: путь сохранения
            fps: Кадров в секунду
        """
        try:
            self.ani.save(save_path + ".gif", writer="pillow", dpi=100, fps=fps)
        except Exception as e:
            print(f"Ошибка при сохранении анимации: {e}")
        plt.close(self.visualizer.fig)


def animation(individual, env, interval_animation=10, device="cpu", count_steps=200, render=True, save_path=None):
    """Демонстрация анимации."""
    print("\n=== Simple Animation ===")
    dog, target = env.reset()

    # Создаем анимацию
    anim = SimpleAnimation(
        env=env,
        dog=dog,
        individual=individual,
        target=target,
        steps=count_steps,
        interval=interval_animation,
        device=device
    )
    if render:
        print("Animation created. Showing window...")
        anim.show()
    if save_path is not None:
        anim.save(save_path)
