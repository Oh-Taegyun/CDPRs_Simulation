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

	// Pin where cables can attach
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Pins")
	UStaticMeshComponent* CablePin;

	// Mass of the actor
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Physics")
	UStaticMeshComponent* PhysicsMesh;

	// 질량 설정, 핀 넘버 설정
	UPROPERTY(EditAnywhere)
	float mass;



	// float PIN_NUMBER;
};
