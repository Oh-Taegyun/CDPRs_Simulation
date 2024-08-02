// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "CableComponent.h"
#include "Components/StaticMeshComponent.h"
#include "PhysicsEngine/PhysicsConstraintComponent.h"
#include "Pulley.generated.h"

class AEndEffector;

UCLASS()
class CDPR_SIMULATION_API APulley : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	APulley();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Component")
	UStaticMeshComponent* Pulley_Mesh;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cable");
	UCableComponent* CableComponent;

	UPROPERTY(VisibleAnywhere)
	UPhysicsConstraintComponent* PhysicsConstraint;

	UPROPERTY(EditAnywhere, Category = "Pulley");
	float MaxCableLength;

	UPROPERTY(EditAnywhere, Category = "Pulley");
	float MinCableLength;

	UPROPERTY(EditAnywhere, Category = "Pulley");
	float MIN_Tension;

	UPROPERTY(EditAnywhere, Category = "Pulley");
	float MAX_Tension;

	UPROPERTY(EditAnywhere, Category = "Pulley");
	int32 Pulley_Number;

	UPROPERTY(EditAnywhere, Category = "Pulley");
	float Tension;

	UPROPERTY(EditAnywhere, Category = "Pulley");
	float CableLength;

	UPROPERTY(EditAnywhere, Category = "Pulley");
	bool PIN_CONNECT;

	AEndEffector* End_Effector;
	USceneComponent* Pin;
	void ApplyCableTension(float input_CableLength, float input_Tension); // 케이블 에 대한 힘 적용
	void AttachCableToEndEffector();
	
};
