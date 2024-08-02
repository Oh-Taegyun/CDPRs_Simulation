// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/StaticMeshComponent.h"
#include "EndEffector.generated.h"



UCLASS()
class CDPR_SIMULATION_API AEndEffector : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AEndEffector();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	// 핀 모아둠
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Pins")
	USceneComponent* Pin_1;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Pins")
	USceneComponent* Pin_2;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Pins")
	USceneComponent* Pin_3;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Pins")
	USceneComponent* Pin_4;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Pins")
	USceneComponent* Pin_5;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Pins")
	USceneComponent* Pin_6;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Pins")
	USceneComponent* Pin_7;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Pins")
	USceneComponent* Pin_8;

	// 물리 메시 설정
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Physics")
	UStaticMeshComponent* PhysicsMesh;
};
