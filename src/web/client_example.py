#!/usr/bin/env python3
"""
Simple WebSocket client example for Regicide game.
This demonstrates how players can connect to your game server.
"""

import asyncio
import websockets
import json
import sys

class RegicideClient:
    def __init__(self, uri="ws://localhost:8765"):
        self.uri = uri
        self.websocket = None
        self.session_id = None
        self.player_name = None
        self.hand = []
        self.game_active = False

    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.uri)
            print(f"Connected to Regicide server at {self.uri}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    async def send_message(self, message):
        """Send a message to the server"""
        if self.websocket:
            await self.websocket.send(json.dumps(message))

    async def create_game(self, player_name, num_players=2):
        """Create a new game session"""
        self.player_name = player_name
        await self.send_message({
            'type': 'create_game',
            'player_name': player_name,
            'num_players': num_players
        })

    async def join_game(self, session_id, player_name):
        """Join an existing game session"""
        self.session_id = session_id
        self.player_name = player_name
        await self.send_message({
            'type': 'join_game',
            'session_id': session_id,
            'player_name': player_name
        })

    async def play_cards(self, card_indices):
        """Play selected cards"""
        await self.send_message({
            'type': 'player_action',
            'action': 'play_cards',
            'card_indices': card_indices
        })

    async def yield_turn(self):
        """Yield the turn"""
        await self.send_message({
            'type': 'player_action',
            'action': 'yield'
        })

    async def handle_server_message(self, message):
        """Handle incoming messages from server"""
        data = json.loads(message)
        msg_type = data.get('type')

        if msg_type == 'welcome':
            print(f"✓ {data['message']}")
            print(f"Client ID: {data['client_id']}")

        elif msg_type == 'game_created':
            self.session_id = data['session_id']
            print(f"✓ Game created! Session ID: {self.session_id}")
            print(f"Waiting for {data['players_needed']} players...")

        elif msg_type == 'player_joined':
            print(f"✓ {data['player_name']} joined the game")
            print(f"Players: {data['players_count']}/{data['players_needed']}")

        elif msg_type == 'game_started':
            print("🎮 Game has started!")
            self.game_active = True

        elif msg_type == 'game_state':
            await self.display_game_state(data)

        elif msg_type == 'action_required':
            if data.get('active_player') and self.game_active:
                await self.handle_action_request(data)

        elif msg_type == 'input_required':
            await self.handle_input_request(data)

        elif msg_type == 'cards_played':
            print(f"🃏 {data['player_name']} played: {', '.join(data['cards'])}")
            print(f"   Total attack: {data['total_attack']}")

        elif msg_type == 'enemy_defeated':
            print(f"💀 {data.get('enemy_name', 'Enemy')} defeated!")

        elif msg_type == 'game_ended':
            result = data['result']
            print(f"\n🎯 GAME OVER: {result.upper()}")
            print(f"   {data['message']}")
            self.game_active = False

        elif msg_type == 'error':
            print(f"❌ Error: {data['message']}")

        else:
            print(f"📨 {msg_type}: {data}")

    async def display_game_state(self, state):
        """Display current game state"""
        enemy = state['current_enemy']
        print(f"\n🏰 Current Enemy: {enemy['name']}")
        print(f"   Health: {enemy['health'] - enemy['damage_taken']}/{enemy['health']}")
        print(f"   Attack: {enemy['attack']}")

        print(f"\n👤 Active Player: {state['active_player']}")

        # Display hand if it's in the state
        for player in state['players']:
            if player['name'] == self.player_name and 'hand' in player:
                self.hand = player['hand']
                print(f"\n🃏 Your Hand:")
                for i, card in enumerate(self.hand):
                    print(f"   [{i}] {card['display_name']} (Value: {card['value']})")

    async def handle_action_request(self, data):
        """Handle action request from server"""
        if data.get('active_player'):
            print(f"\n⚡ Your turn! Choose action:")
            print("   [1] Play cards")
            print("   [2] Yield turn")

            choice = input("Enter choice (1-2): ").strip()

            if choice == "1":
                await self.select_and_play_cards()
            elif choice == "2":
                await self.yield_turn()
            else:
                print("Invalid choice, yielding turn...")
                await self.yield_turn()

    async def handle_input_request(self, data):
        """Handle specific input requests"""
        if data.get('target_client') or data.get('player_name') == self.player_name:
            input_type = data.get('input_type')

            if input_type == 'card_selection':
                await self.select_and_play_cards()
            elif input_type == 'discard_selection':
                await self.select_discard_cards(data.get('required_value', 0))

    async def select_and_play_cards(self):
        """Interactive card selection for playing"""
        if not self.hand:
            print("No cards in hand!")
            await self.yield_turn()
            return

        print("\nSelect cards to play (space-separated indices) or 'y' to yield:")
        indices_input = input("> ").strip().lower()

        if indices_input == 'y':
            await self.yield_turn()
            return

        try:
            indices = [int(x) for x in indices_input.split()]
            if all(0 <= idx < len(self.hand) for idx in indices):
                await self.play_cards(indices)
            else:
                print("Invalid card indices!")
                await self.yield_turn()
        except ValueError:
            print("Invalid input!")
            await self.yield_turn()

    async def select_discard_cards(self, required_value):
        """Interactive card selection for discarding"""
        print(f"\n💥 You must discard cards worth at least {required_value} to defend!")
        print("Select cards to discard (space-separated indices):")

        indices_input = input("> ").strip()

        try:
            indices = [int(x) for x in indices_input.split()]
            if all(0 <= idx < len(self.hand) for idx in indices):
                total_value = sum(self.hand[idx]['value'] for idx in indices)
                if total_value >= required_value:
                    await self.send_message({
                        'type': 'player_action',
                        'action': 'discard_cards',
                        'card_indices': indices
                    })
                else:
                    print(f"Not enough value! Need {required_value}, selected {total_value}")
            else:
                print("Invalid card indices!")
        except ValueError:
            print("Invalid input!")

    async def listen(self):
        """Listen for messages from server"""
        try:
            async for message in self.websocket:
                await self.handle_server_message(message)
        except websockets.exceptions.ConnectionClosed:
            print("Connection to server lost.")
        except Exception as e:
            print(f"Error: {e}")

    async def run(self):
        """Main client loop"""
        if not await self.connect():
            return

        print("\n🎴 Welcome to Regicide!")
        print("Commands:")
        print("  create <name> [players] - Create new game")
        print("  join <session_id> <name> - Join existing game")
        print("  quit - Exit")

        # Start listening in background
        listen_task = asyncio.create_task(self.listen())

        try:
            while True:
                if not self.game_active:
                    command = input("\n> ").strip().split()

                    if not command:
                        continue

                    if command[0] == "create":
                        name = command[1] if len(command) > 1 else "Player"
                        players = int(command[2]) if len(command) > 2 else 2
                        await self.create_game(name, players)

                    elif command[0] == "join":
                        if len(command) >= 3:
                            session_id = command[1]
                            name = command[2]
                            await self.join_game(session_id, name)
                        else:
                            print("Usage: join <session_id> <name>")

                    elif command[0] == "quit":
                        break

                    else:
                        print("Unknown command. Use: create, join, or quit")

                else:
                    # Game is active, just wait
                    await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\nGoodbye!")
        finally:
            listen_task.cancel()
            if self.websocket:
                await self.websocket.close()

async def main():
    client = RegicideClient()
    await client.run()

if __name__ == "__main__":
    asyncio.run(main())
