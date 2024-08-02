// Fill out your copyright notice in the Description page of Project Settings.


#include "Pulley_ver_bp.h"
#include "Kismet/GameplayStatics.h"
#include "GameFramework/Actor.h"
#include "Engine/World.h"
#include "Components/StaticMeshComponent.h"

// Sets default values
APulley_ver_bp::APulley_ver_bp()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

    Pulley_Mesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Pulley"));
    RootComponent = Pulley_Mesh;

    CableComponent = CreateDefaultSubobject<UCableComponent>(TEXT("CableComponent"));
    CableComponent->SetupAttachment(Pulley_Mesh);

    PhysicsConstraint = CreateDefaultSubobject<UPhysicsConstraintComponent>(TEXT("PhysicsConstraint"));
    PhysicsConstraint->SetupAttachment(Pulley_Mesh);

    MaxCableLength = 1000.0f; // Set a default maximum cable length
    Tension = 1000.0f;
    Pulley_Number = 0;
    MaxCableLength = CableLength;
}

// Called when the game starts or when spawned
void APulley_ver_bp::BeginPlay()
{
	Super::BeginPlay();
    Get_EndEffector(Pulley_Number);
    
}

// Called every frame
void APulley_ver_bp::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

    ApplyCableTension(CableLength, Tension);

}


void APulley_ver_bp::Get_EndEffector(int32 num) {
    // 씬에 있는 모든 Actor B를 찾습니다.
    TArray<AActor*> FoundActors;
    UGameplayStatics::GetAllActorsWithTag(GetWorld(), FName("BP_EndEffector"), FoundActors);

    for (AActor* Actor : FoundActors)
    {
        // ComponentNumber에 따른 Component 이름 결정
        FString ComponentName;
        switch (num)
        {
        case 1:
            ComponentName = "Pin_1";
            break;
        case 2:
            ComponentName = "Pin_2";
            break;
        case 3:
            ComponentName = "Pin_3";
            break;
        case 4:
            ComponentName = "Pin_4";
            break;
        case 5:
            ComponentName = "Pin_5";
            break;
        case 6:
            ComponentName = "Pin_6";
            break;
        case 7:
            ComponentName = "Pin_7";
            break;
        case 8:
            ComponentName = "Pin_8";
            break;
        default:
            UE_LOG(LogTemp, Warning, TEXT("Invalid Component Number"));
            return;
        }

        // Actor B에서 모든 Scene Component를 가져옵니다.
        TArray<USceneComponent*> SceneComponents;
        Actor->GetComponents(SceneComponents); // 이 부분이 빠져 있었습니다.

        for (USceneComponent* Component : SceneComponents)
        {
            if (Component && Component->GetName() == ComponentName)
            {
                Pin = Component;
                CableComponent->SetAttachEndToComponent(Pin, Pin->GetFName());

                if (GEngine)
                {
                    GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, FString::Printf(TEXT("Found Scene Component: %s"), *Pin->GetName()));
                }

                return;
            }
        }
    }

    UE_LOG(LogTemp, Warning, TEXT("Scene Component with specified name not found"));
}


void APulley_ver_bp::ApplyCableTension(float input_CableLength, float input_Tension) {
    // CableComponent, Pin, End_Effector가 모두 유효한지 확인합니다.
    if (CableComponent && Pin && End_Effector)
    {
        // 최소 케이블 길이를 설정합니다 (예: 1.0cm).
        float MinCableLength = 1.0f;

        // 케이블의 시작 위치를 가져옵니다.
        FVector StartLocation = CableComponent->GetComponentLocation();

        // 케이블의 끝 위치를 가져옵니다.
        FVector EndLocation = Pin->GetComponentLocation();

        // 시작 위치와 끝 위치 사이의 방향 벡터를 계산하고, 정규화합니다.
        FVector Direction = (EndLocation - StartLocation).GetSafeNormal();

        // 현재 케이블 길이를 계산합니다.
        float CurrentLength = (EndLocation - StartLocation).Size();

        // 케이블 길이가 입력된 케이블 길이와 다른 경우 처리합니다.
        if (CurrentLength != input_CableLength)
        {
            // 최소 케이블 길이보다 작은 경우 최소 길이로 설정합니다.
            float AdjustedDesiredLength = FMath::Max(input_CableLength, MinCableLength);

            // 조정된 케이블 길이에 맞춰 클램프된 끝 위치를 계산합니다.
            FVector ClampedEndLocation = StartLocation + Direction * AdjustedDesiredLength;

            // 핀과 엔드 이펙터 간의 오프셋을 계산합니다.
            FVector PinToEndEffectorOffset = End_Effector->GetComponentLocation() - Pin->GetComponentLocation();

            // 클램프된 엔드 이펙터 위치를 계산합니다.
            FVector ClampedEndEffectorLocation = ClampedEndLocation + PinToEndEffectorOffset;

            // 엔드 이펙터가 이동할 방향을 정규화된 벡터로 계산합니다.
            FVector ForceDirection = (ClampedEndEffectorLocation - End_Effector->GetComponentLocation()).GetSafeNormal();

            // 힘의 크기를 계산합니다.
            float ForceStrength = (ClampedEndEffectorLocation - End_Effector->GetComponentLocation()).Size() * input_Tension;

            // 엔드 이펙터에 힘을 가합니다.
            End_Effector->AddForce(ForceDirection * ForceStrength);

            // 케이블 컴포넌트의 렌더 상태를 갱신합니다.
            CableComponent->MarkRenderStateDirty();
        }
    }
}


