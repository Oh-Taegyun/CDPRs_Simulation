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

	// 풀리 메시 설정
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Component")
	UStaticMeshComponent* Pulley_Mesh;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Cable");
	UCableComponent* CableComponent;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Constraints")
	UPhysicsConstraintComponent* CableConstraint;

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly);
	float MaxCableLength;

	UPROPERTY(EditDefaultsOnly, BlueprintReadOnly);
	float Tension;

private:
	AEndEffector* EndEffector;
	void AttachCableToComponent();
	void LimitCableLength();
	void SetCableLength(float NewLength);
	void ApplyCableTension(float TensionStrength);
	void CutCable();
};
