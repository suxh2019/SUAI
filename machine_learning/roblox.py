import random
import time
from dataclasses import dataclass, field
from typing import List


# ==========================================
# ROBLOX RIVALS INSPIRED PYTHON GAME
# ==========================================
# Features:
# - Classes with abilities
# - Critical hits
# - Energy system
# - Dodging
# - Inventory
# - Enemy AI
# - Combo attacks
# - Boss fights
# - Leveling system
# - Turn-based battle engine
# ==========================================


@dataclass
class Ability:
    name: str
    damage: int
    energy_cost: int
    crit_chance: float
    special_effect: str

    def use(self, attacker, target):
        if attacker.energy < self.energy_cost:
            print(f"⚠️ {attacker.name} does not have enough energy for {self.name}!")
            return

        attacker.energy -= self.energy_cost

        dodge_roll = random.random()
        if dodge_roll < target.dodge_chance:
            print(f"💨 {target.name} dodged the attack!")
            return

        base_damage = self.damage + attacker.attack_power

        crit_roll = random.random()
        critical = crit_roll < self.crit_chance

        if critical:
            base_damage *= 2
            print("🔥 CRITICAL HIT!")

        shield_block = min(target.shield, base_damage)
        remaining_damage = base_damage - shield_block

        target.shield -= shield_block
        target.health -= remaining_damage

        print(f"⚔️ {attacker.name} used {self.name}!")
        print(f"💥 Damage: {remaining_damage}")

        if shield_block > 0:
            print(f"🛡️ Shield blocked {shield_block} damage!")

        if self.special_effect == "burn":
            target.status_effects.append(("burn", 3))
            print(f"🔥 {target.name} is burning!")

        if self.special_effect == "freeze":
            target.status_effects.append(("freeze", 1))
            print(f"❄️ {target.name} is frozen!")


@dataclass
class Character:
    name: str
    max_health: int
    health: int
    energy: int
    attack_power: int
    dodge_chance: float
    shield: int
    level: int
    xp: int
    inventory: List[str] = field(default_factory=list)
    abilities: List[Ability] = field(default_factory=list)
    status_effects: List[tuple] = field(default_factory=list)

    def show_stats(self):
        print("\n========================")
        print(f"🏷️ Name: {self.name}")
        print(f"❤️ Health: {self.health}/{self.max_health}")
        print(f"⚡ Energy: {self.energy}")
        print(f"⚔️ Attack: {self.attack_power}")
        print(f"🛡️ Shield: {self.shield}")
        print(f"🎯 Dodge Chance: {self.dodge_chance * 100:.0f}%")
        print(f"⭐ Level: {self.level}")
        print(f"✨ XP: {self.xp}")
        print(f"🎒 Inventory: {self.inventory}")
        print("========================\n")

    def process_effects(self):
        new_effects = []

        for effect, turns in self.status_effects:
            if effect == "burn":
                self.health -= 5
                print(f"🔥 {self.name} takes 5 burn damage!")

            elif effect == "freeze":
                print(f"❄️ {self.name} is frozen and loses a turn!")

            if turns - 1 > 0:
                new_effects.append((effect, turns - 1))

        self.status_effects = new_effects

    def level_up(self):
        required_xp = self.level * 100

        while self.xp >= required_xp:
            self.level += 1
            self.max_health += 20
            self.health = self.max_health
            self.attack_power += 5
            self.energy += 10
            self.shield += 5
            self.xp -= required_xp

            print(f"\n🎉 {self.name} LEVELED UP TO LEVEL {self.level}! 🎉")
            print("💪 Stats increased!")

            required_xp = self.level * 100

    def basic_attack(self, target):
        damage = self.attack_power + random.randint(1, 10)

        dodge_roll = random.random()
        if dodge_roll < target.dodge_chance:
            print(f"💨 {target.name} dodged the attack!")
            return

        shield_block = min(target.shield, damage)
        remaining_damage = damage - shield_block

        target.shield -= shield_block
        target.health -= remaining_damage

        print(f"👊 {self.name} attacks {target.name}!")
        print(f"💥 Damage dealt: {remaining_damage}")

    def use_item(self, item_name):
        if item_name not in self.inventory:
            print("❌ Item not found!")
            return

        if item_name == "Health Potion":
            heal = 30
            self.health = min(self.max_health, self.health + heal)
            print(f"🧪 {self.name} healed for {heal} HP!")

        elif item_name == "Energy Core":
            self.energy += 25
            print(f"⚡ {self.name} restored 25 energy!")

        elif item_name == "Shield Booster":
            self.shield += 20
            print(f"🛡️ {self.name} gained 20 shield!")

        self.inventory.remove(item_name)


class EnemyAI:

    @staticmethod
    def choose_action(enemy, player):
        available_abilities = [
            ability for ability in enemy.abilities
            if enemy.energy >= ability.energy_cost
        ]

        if available_abilities and random.random() > 0.4:
            ability = random.choice(available_abilities)
            ability.use(enemy, player)
        else:
            enemy.basic_attack(player)


class BattleArena:

    def __init__(self, player, enemy):
        self.player = player
        self.enemy = enemy

    def player_turn(self):
        print("\n===== YOUR TURN =====")
        self.player.show_stats()
        self.enemy.show_stats()

        print("1. Basic Attack")
        print("2. Use Ability")
        print("3. Use Item")

        choice = input("Choose action: ")

        if choice == "1":
            self.player.basic_attack(self.enemy)

        elif choice == "2":
            print("\nAbilities:")

            for index, ability in enumerate(self.player.abilities):
                print(
                    f"{index + 1}. {ability.name} "
                    f"(Damage: {ability.damage}, "
                    f"Energy: {ability.energy_cost})"
                )

            ability_choice = int(input("Choose ability: ")) - 1

            if 0 <= ability_choice < len(self.player.abilities):
                ability = self.player.abilities[ability_choice]
                ability.use(self.player, self.enemy)

        elif choice == "3":
            print(f"Inventory: {self.player.inventory}")
            item = input("Enter item name: ")
            self.player.use_item(item)

    def enemy_turn(self):
        print("\n===== ENEMY TURN =====")
        EnemyAI.choose_action(self.enemy, self.player)

    def battle_loop(self):

        print("\n🔥 BATTLE START! 🔥")

        while self.player.health > 0 and self.enemy.health > 0:

            self.player.process_effects()
            self.enemy.process_effects()

            if self.player.health <= 0 or self.enemy.health <= 0:
                break

            self.player_turn()

            if self.enemy.health <= 0:
                break

            time.sleep(1)

            self.enemy_turn()

            time.sleep(1)

        if self.player.health > 0:
            print(f"\n🏆 {self.player.name} WINS!")

            xp_gain = random.randint(80, 150)
            self.player.xp += xp_gain

            print(f"✨ Gained {xp_gain} XP!")

            loot_table = [
                "Health Potion",
                "Energy Core",
                "Shield Booster"
            ]

            loot = random.choice(loot_table)
            self.player.inventory.append(loot)

            print(f"🎁 You found: {loot}")

            self.player.level_up()

        else:
            print(f"\n💀 {self.enemy.name} WINS!")


# ==========================================
# CREATE ABILITIES
# ==========================================

fireball = Ability(
    name="Fireball",
    damage=25,
    energy_cost=15,
    crit_chance=0.25,
    special_effect="burn"
)

ice_strike = Ability(
    name="Ice Strike",
    damage=18,
    energy_cost=12,
    crit_chance=0.15,
    special_effect="freeze"
)

shadow_blade = Ability(
    name="Shadow Blade",
    damage=35,
    energy_cost=20,
    crit_chance=0.35,
    special_effect=""
)

mega_smash = Ability(
    name="Mega Smash",
    damage=40,
    energy_cost=25,
    crit_chance=0.20,
    special_effect=""
)


# ==========================================
# CREATE PLAYER
# ==========================================

player = Character(
    name="Cyber Ninja",
    max_health=120,
    health=120,
    energy=60,
    attack_power=12,
    dodge_chance=0.15,
    shield=20,
    level=1,
    xp=0,
    inventory=["Health Potion", "Energy Core"],
    abilities=[fireball, ice_strike, shadow_blade]
)


# ==========================================
# CREATE BOSS ENEMY
# ==========================================

boss = Character(
    name="Titan Destroyer",
    max_health=180,
    health=180,
    energy=80,
    attack_power=15,
    dodge_chance=0.10,
    shield=25,
    level=3,
    xp=0,
    inventory=[],
    abilities=[mega_smash, fireball]
)


# ==========================================
# START GAME
# ==========================================

print("""
██████╗  ██████╗ ██████╗ ██╗      ██████╗ ██╗  ██╗
██╔══██╗██╔═══██╗██╔══██╗██║     ██╔═══██╗╚██╗██╔╝
██████╔╝██║   ██║██████╔╝██║     ██║   ██║ ╚███╔╝
██╔══██╗██║   ██║██╔══██╗██║     ██║   ██║ ██╔██╗
██║  ██║╚██████╔╝██████╔╝███████╗╚██████╔╝██╔╝ ██╗
╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝
""")

print("🎮 ROBLOX RIVALS PYTHON EDITION 🎮")
print("⚔️ Prepare for battle! ⚔️")

arena = BattleArena(player, boss)
arena.battle_loop()

print("\n🎯 GAME OVER")
