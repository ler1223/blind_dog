import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import BasicTrain
import matplotlib.animation


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
        self.visualizer = MinimalVisualizer(field_size=env.field_size)

        # Для хранения истории
        self.history = {
            'positions': [],
            'fitness': []
        }

        # Создаем анимацию
        self.ani = FuncAnimation(
            self.visualizer.fig,
            self._update_frame,
            frames=steps,
            interval=interval,
            repeat=False,
            blit=False
        )

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
        self.dog, self.target, reward = self.env.step(self.dog, self.target, accelerate, angle_acceleration)

        # Обновляем визуализацию
        self.visualizer.update_scene(
            self.dog, self.target,
            step=frame,
            fitness=reward
        )

        # Сохраняем историю
        self.history['positions'].append(self.dog.position.numpy())
        self.history['fitness'].append(reward)

        if frame == self.steps - 1:
            plt.close(self.visualizer.fig)

    def show(self):
        """Показывает анимацию."""
        self.visualizer.show()

    def save(self, save_path, fps=30):
        """
        Сохраняет анимацию в файл.
        Args:
            save_path: путь сохранения
            fps: Кадров в секунду
        """
        self.ani.save(save_path + ".gif", writer="pillow", dpi=100, fps=fps)
        plt.close(self.visualizer.fig)


def animation(individual, env, interval_animation=10, device="cpu", count_steps=200, render=True, save_path=None):
    """Демонстрация анимации."""
    print("\n=== Simple Animation Demo ===")
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
