"""Lobby regression test: the game CREATOR must receive their own join.

Root cause (orchestrator-verified): ``handle_create_game`` sent
``game_created`` to the creator but never broadcast ``player_joined``, so
the creator's client waiting-room list stayed empty ("0 / N heroes
assembled") while joiners saw correct counts. The fix mirrors the join
path's broadcast shape (app.py ``handle_join_game``) right after
``game_created``.

Reuses the proven ephemeral-port integration helpers from
``test_integration.py`` (same directory, pytest rootdir sys.path).
"""
import asyncio

import test_integration as ti


async def _creator_receives_own_join():
    server, srv, port = await ti.start_ephemeral_server()
    client = None
    try:
        client = await ti.connect_client(port)
        await client.send({
            "type": "create_game",
            "player_name": "Finn",
            "num_players": 2,
        })

        created = await client.recv_until("game_created")
        assert created["session_id"]  # non-empty room id
        assert created["player_name"] == "Finn"
        assert created["players_needed"] == 2

        # The fix under test: creator's own join broadcast, same shape as
        # the join path (handle_join_game).
        joined = await client.recv_until("player_joined")
        assert joined["player_name"] == "Finn"
        assert joined["players_count"] == 1
        assert joined["players_needed"] == 2
    finally:
        if client is not None:
            await ti.shutdown_server(srv, [client])
        else:
            srv.close()
            await srv.wait_closed()


def test_creator_receives_own_join_broadcast():
    """Creator gets game_created followed by their own player_joined."""
    asyncio.run(_creator_receives_own_join())
