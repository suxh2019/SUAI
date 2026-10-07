import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRAVITY = 0.8
PLAYER_SPEED = 5
JUMP_SPEED = -15
FPS = 60

# Colors
WHITE = (255, 255, 255)
SKY_BLUE = (135, 206, 235)
BROWN = (139, 69, 19)
RED = (220, 20, 60)
GREEN = (34, 139, 34)

# Screen setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Simple Mario Game")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((32, 48))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = 100
        self.rect.y = SCREEN_HEIGHT - 100
        self.vel_x = 0
        self.vel_y = 0
        self.on_ground = False
        self.score = 0

    def update(self, platforms):
        self.vel_y += GRAVITY
        if self.vel_y > 12:
            self.vel_y = 12

        # Horizontal movement + collision
        self.rect.x += self.vel_x
        self.collide(self.vel_x, 0, platforms)

        # Vertical movement + collision
        self.rect.y += self.vel_y
        self.on_ground = False
        self.collide(0, self.vel_y, platforms)

        # Screen boundaries
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH * 2:  # Allow some scrolling feel
            self.rect.right = SCREEN_WIDTH * 2

    def collide(self, vel_x, vel_y, platforms):
        for p in platforms:
            if pygame.sprite.collide_rect(self, p):
                if vel_x > 0:
                    self.rect.right = p.rect.left
                elif vel_x < 0:
                    self.rect.left = p.rect.right
                if vel_y > 0:
                    self.rect.bottom = p.rect.top
                    self.vel_y = 0
                    self.on_ground = True
                elif vel_y < 0:
                    self.rect.top = p.rect.bottom
                    self.vel_y = 0

    def jump(self):
        if self.on_ground:
            self.vel_y = JUMP_SPEED
            self.on_ground = False


class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, w, h, color=BROWN):
        super().__init__()
        self.image = pygame.Surface((w, h))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

# Enemy class
class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((28, 28))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        # control the speed of enemy -2: slow. -1 would be very slow
        self.vel_x = -1

    def update(self, platforms):
        self.rect.x += self.vel_x
        for p in platforms:
            if pygame.sprite.collide_rect(self, p):
                self.vel_x *= -1
                break
        if self.rect.left < 50 or self.rect.right > SCREEN_WIDTH + 100:
            self.vel_x *= -1


# Sprite groups
all_sprites = pygame.sprite.Group()
platforms = pygame.sprite.Group()
enemies = pygame.sprite.Group()

# Player
player = Player()
all_sprites.add(player)

# Platforms (ground + floating)
ground = Platform(0, SCREEN_HEIGHT - 40, SCREEN_WIDTH * 3, 40)
all_sprites.add(ground)
platforms.add(ground)

for pos in [(250, 450), (500, 350), (750, 250), (1000, 400)]:
    p = Platform(pos[0], pos[1], 160, 20)
    all_sprites.add(p)
    platforms.add(p)

# Enemies
enemy1 = Enemy(450, SCREEN_HEIGHT - 70)
enemy2 = Enemy(850, SCREEN_HEIGHT - 70)
all_sprites.add(enemy1, enemy2)
enemies.add(enemy1, enemy2)

# Game variables
camera_x = 0
running = True

while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                player.jump()

    # Controls
    keys = pygame.key.get_pressed()
    player.vel_x = 0
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        player.vel_x = -PLAYER_SPEED
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        player.vel_x = PLAYER_SPEED

    # Update everything
    all_sprites.update(platforms)

    # Camera follows player (simple scrolling)
    if player.rect.x - camera_x > SCREEN_WIDTH // 2:
        camera_x = player.rect.x - SCREEN_WIDTH // 2

    # Collisions
    if pygame.sprite.spritecollide(player, enemies, False):
        print("💥 Game Over! You got hit!")
        running = False

    # Win condition
    if player.rect.x > 1400:
        player.score += 500
        print(f"🎉 You Win! Final Score: {player.score}")
        running = False

    # Drawing
    screen.fill(SKY_BLUE)

    # Draw with camera offset
    for sprite in all_sprites:
        screen.blit(sprite.image, (sprite.rect.x - camera_x, sprite.rect.y))

    # HUD
    score_text = font.render(f"Score: {player.score}", True, WHITE)
    screen.blit(score_text, (10, 10))
    controls_text = font.render("← → / A D : Move    SPACE : Jump", True, WHITE)
    screen.blit(controls_text, (10, SCREEN_HEIGHT - 30))

    pygame.display.flip()

pygame.quit()
sys.exit()