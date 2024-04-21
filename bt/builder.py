import os
from bt.actions import *
from bt.nodes_rl import *
from bt.conditions import *
from bt.nodes_test import *
from rl import RLBuilder

BASE_DIR = os.path.dirname(__file__)


class BTBuilder(RLBuilder):
    def register_default(self):
        super().register_default()

        self.register_node(
                TurnLeft,
                TurnRight,
                MoveForward,
                ManualControl,
                OpenDoor,
                PickUpKey,
                MoveLeft,
                MoveUp,
                MoveDown,
                MoveToPosition,
                MoveToObject,
                ApproachObject,
                MoveToGoal,
                ApproachKey,
                ApproachDoor,
                ExploreUnseen,
                TurnToObject,
                TurnToKey,
                TurnToDoor
        )

        self.register_node(
                IsObjectFound,
                IsUnseenFound,
                IsGoalFound,
                IsDoorFound,
                IsKeyFound,
                IsBallFound,
                IsBoxFound,
                CanMoveToObject,
                CanMoveToGoal,
                CanMoveToUnseen,
                CanApproachDoor,
                CanApproachKey,
                IsReachObject,
                IsReachGoal,
                IsNearObject,
                IsNearGoal,
                IsNearDoor,
                IsNearKey,
                IsNearBall,
                IsNearBox,
                IsKeyHeld,
                IsObjectInFront,
                IsKeyInFront,
                IsDoorInFront,
                IsDoorOpen,
                IsDoorClosed,
                IsDoorLocked,
                Toggle,
                Pickup,
                Drop
        )

        self.register_node(
                RLSwitcher,
                RLSelector,
                RLSequence,
                RLCondition,
                RLIntValue,
                RLFloatValue,
                RLFloatArrayValue,
                RLReward,
                RLAction
        )

        self.register_node(
                MakeTrouble,
                ToggleTargetColor
        )
