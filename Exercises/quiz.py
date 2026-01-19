import matplotlib.pyplot as plt
import random
from io import BytesIO
import numpy as np
import pygame
import matplotlib
matplotlib.use("Agg")

# Konfiguration


WIDTH, HEIGHT = 1000, 700
ROUNDS = 8
SAMPLE_SIZE = 600

PLOT_AREA = pygame.Rect(20, 170, 480, 510)

BG = (20, 20, 24)
PANEL = (35, 35, 42)
WHITE = (240, 240, 240)
MUTED = (180, 180, 190)
GREEN = (60, 200, 120)
RED = (230, 80, 80)
BTN = (55, 55, 70)
BTN_HOVER = (75, 75, 95)

# Distributioner (slumpade parametrar)


def gen_normal(rng):
    mu = rng.uniform(-1.5, 1.5)
    sigma = rng.uniform(0.6, 2.0)
    x = rng.normal(loc=mu, scale=sigma, size=SAMPLE_SIZE)
    return "Normal", x


def gen_uniform(rng):
    a = rng.uniform(-3, 1)
    b = a + rng.uniform(2, 6)
    x = rng.uniform(low=a, high=b, size=SAMPLE_SIZE)
    return "Uniform", x


def gen_binomial(rng):
    n = int(rng.integers(8, 25))
    p = float(rng.uniform(0.15, 0.7))
    x = rng.binomial(n=n, p=p, size=SAMPLE_SIZE)
    return "Binomial", x


def gen_neg_binomial(rng):
    n = int(rng.integers(3, 15))
    p = float(rng.uniform(0.2, 0.6))
    x = rng.negative_binomial(n=n, p=p, size=SAMPLE_SIZE)
    return "Negative binomial", x


def gen_gamma(rng):
    shape = float(rng.uniform(0.8, 4.0))
    scale = float(rng.uniform(0.4, 2.0))
    x = rng.gamma(shape=shape, scale=scale, size=SAMPLE_SIZE)
    return "Gamma", x


def gen_geometric(rng):
    p = float(rng.uniform(0.15, 0.6))
    x = rng.geometric(p=p, size=SAMPLE_SIZE)
    return "Geometric", x


GENERATORS = [
    gen_normal,
    gen_uniform,
    gen_binomial,
    gen_neg_binomial,
    gen_gamma,
    gen_geometric
]

ALL_NAMES = ["Normal", "Uniform", "Binomial",
             "Negative binomial", "Gamma", "Geometric"]

# Plot - pygame image


def make_plot_image(data: np.ndarray, mode: str) -> pygame.Surface:
    """
    mode: "hist" eller "ogive"
    Returnerar en pygame.Surface med plottad bild.
    """
    fig = plt.figure(figsize=(7.8, 4.6), dpi=140)
    ax = fig.add_subplot(111)

    if mode == "hist":
        ax.hist(data, bins=30, edgecolor="black")
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Value")
        ax.set_title("Identify the distribution (Histogram)")
    else:
        xs = np.sort(data)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Cumlative probability")
        ax.set_xlabel("Value")
        ax.set_title("Identify the distribution (Ogive)")

    fig.tight_layout()

    bio = BytesIO()
    fig.savefig(bio, format="png")
    plt.close(fig)
    bio.seek(0)

    img = pygame.image.load(bio, "plot.png").convert_alpha()
    return img

# UI-hjälpare


def draw_text(screen, text, font, color, x, y):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))
    return surf.get_rect(topleft=(x, y))


def draw_button(screen, rect, text, font, mouse_pos):
    color = BTN_HOVER if rect.collidepoint(mouse_pos) else BTN
    pygame.draw.rect(screen, color, rect, border_radius=10)
    pygame.draw.rect(screen, (110, 110, 135), rect, width=2, border_radius=10)
    label = font.render(text, True, WHITE)
    label_rect = label.get_rect(center=rect.center)
    screen.blit(label, label_rect)

# Spelets huvudlogik


def new_round(rng):
    mode = random.choice(["hist", "ogive"])
    name, data = random.choice(GENERATORS)(rng)

    # 4 alternativ: 1 korrekt + 3 fel
    wrong = [n for n in ALL_NAMES if n != name]
    options = random.sample(wrong, 3) + [name]
    random.shuffle(options)

    plot_surface = make_plot_image(data, mode)
    return name, options, plot_surface, mode


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Distribution Quiz (pygame)")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont(None, 42)
    font = pygame.font.SysFont(None, 30)
    font_small = pygame.font.SysFont(None, 24)

    rng = np.random.default_rng()

    score = 0
    round_idx = 1
    feedback = None
    feedback_timer = 0

    correct_name, options, plot_img, mode = new_round(rng)

    # Knappar
    btn_w, btn_h = 420, 60
    btn_x = 520
    btn_y0 = 260
    gap = 18
    buttons = []
    for i in range(4):
        rect = pygame.Rect(btn_x, btn_y0 + i * (btn_h + gap), btn_w, btn_h)
        buttons.append(rect)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, rect in enumerate(buttons):
                    if rect.collidepoint(mouse_pos):
                        chosen = options[i]
                        if chosen == correct_name:
                            score += 1
                            feedback = ("Correct!", GREEN)
                        else:
                            feedback = (f"Wrong! Correct: {correct_name}", RED)

                        feedback_timer = pygame.time.get_ticks()

                        # nästa runda (om kvar)
                        if round_idx < ROUNDS:
                            round_idx += 1
                            correct_name, options, plot_img, mode = new_round(
                                rng)
                        else:
                            # avsluta med sista feedbacken kvar på skärm
                            pass
        # Bakgrund
        screen.fill(BG)
        pygame.draw.rect(screen, PANEL, pygame.Rect(
            20, 20, 480, 660), border_radius=14)
        pygame.draw.rect(screen, PANEL, pygame.Rect(
            520, 20, 460, 660), border_radius=14)

        # Header
        draw_text(screen, "Exercise 2 Quiz", font_big, WHITE, 30, 30)
        draw_text(
            screen, f"Round: {min(round_idx, ROUNDS)}/{ROUNDS}", font, MUTED, 30, 80)
        draw_text(screen, f"Score: {score}", font, MUTED, 30, 115)
        draw_text(screen, "Click the correct distribution:",
                  font, WHITE, 540, 200)

        # Visa plotbild
        # (centrera i vänster panelen)
        pygame.draw.rect(screen, (245, 245, 245), PLOT_AREA, border_radius=10)
        plot_scaled = pygame.transform.smoothscale(
            plot_img, (PLOT_AREA.width - 20, PLOT_AREA.height - 20))
        screen.blit(plot_scaled, (PLOT_AREA.x + 10, PLOT_AREA.y + 10))

        # Knappar med alternativ
        for i, rect in enumerate(buttons):
            draw_button(
                screen, rect, f"{chr(65+i)}: {options[i]}", font, mouse_pos)

        # Feedback
        if feedback is not None:
            # Visa feedback i 2.5 sek, men låt den synas även på sista rundan
            now = pygame.time.get_ticks()
            if now - feedback_timer < 2500 or round_idx >= ROUNDS:
                draw_text(screen, feedback[0], font, feedback[1], 540, 520)
            else:
                feedback = None

        # Sluttext om klar
        if round_idx >= ROUNDS:
            draw_text(screen, "Quiz finished! Close the window to exit.",
                      font_small, MUTED, 540, 570)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
