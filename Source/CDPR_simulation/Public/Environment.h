// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "Pulley.h"
#include "EndEffector.h"
#include "Environment.generated.h"

/**
 * 
 */
UCLASS()
class CDPR_SIMULATION_API AEnvironment : public AGameModeBase
{
	GENERATED_BODY()

    public:
        AEnvironment();

    protected:
        virtual void BeginPlay() override;

    public:
        virtual void Tick(float DeltaTime) override;

        UFUNCTION(BlueprintCallable, Category = "Reinforcement Learning")
        void ResetEnvironment(); // 환경 리셋

        UFUNCTION(BlueprintCallable, Category = "Reinforcement Learning")
        float GetReward(const FVector& CurrentPosition, const FRotator& CurrentRotation, float TimeStep);

        UPROPERTY(EditAnywhere)
        TSubclassOf<AActor> BP_Pulley;

        UPROPERTY(EditAnywhere)
        TSubclassOf<AActor> BP_EndEffector;

private:
    FVector SpawnPulleyOnCircle(float Radius, float Higth, int32 Numplley);
    void Spawn_Pulley();
    TArray<FVector> TargetPositions;
    float Reward;
    float MaxDeviation;
    bool done;
    void step(TArray<float> input_CableLength, TArray<float> input_Tension, float DeltaTime);
    void RestartCurrentLevel();
};

