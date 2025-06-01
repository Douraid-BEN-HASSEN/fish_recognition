using UnityEngine;

public class Poisson : MonoBehaviour
{
    public GameObject aquarium;

    [Header("Dimensions souhaitées du poisson (en unités)")]
    public float longueurZ = 1.0f;
    public float hauteurY = 0.4f;
    public float largeurX = 0.3f;

    public float vitesse = 1f;
    public float tempsMinDirection = 1f;
    public float tempsMaxDirection = 3f;

    private Vector3 direction;
    private float tempsRestant;
    private Bounds zoneDeNage;

    public bool isReplay = false;
    void Start()
    {
        //AjusterTaillePoisson();
        if (!isReplay)
        {
            AjusterTaillePoisson();
            CalculerZoneDeNage();
            NouvelleDirection();
        }
    }

    void Update()
    {
        Vector3 tentativePosition = transform.position + direction * vitesse * Time.deltaTime;

        // Si la nouvelle position dépasserait les limites, on inverse la direction correspondante
        if (!zoneDeNage.Contains(tentativePosition))
        {
            Vector3 clamped = tentativePosition;

            if (tentativePosition.x < zoneDeNage.min.x || tentativePosition.x > zoneDeNage.max.x)
                direction.x *= -1;

            if (tentativePosition.y < zoneDeNage.min.y || tentativePosition.y > zoneDeNage.max.y)
                direction.y *= -1;

            if (tentativePosition.z < zoneDeNage.min.z || tentativePosition.z > zoneDeNage.max.z)
                direction.z *= -1;

            NouvelleDirection(); // Optionnel : redéfinir une nouvelle direction après rebond
        }
        else
        {
            transform.position = tentativePosition;
        }

        // Rotation vers la direction du mouvement
        if (direction != Vector3.zero)
        {
            Quaternion rotationCible = Quaternion.LookRotation(direction);
            Vector3 euler = rotationCible.eulerAngles;
            euler.z = 0f;
            transform.rotation = Quaternion.Slerp(transform.rotation, Quaternion.Euler(euler), Time.deltaTime * 2f);
        }

        tempsRestant -= Time.deltaTime;
        if (tempsRestant <= 0)
        {
            NouvelleDirection();
        }
    }

    void NouvelleDirection()
    {
        direction = new Vector3(
            Random.Range(-1f, 1f),
            Random.Range(-0.5f, 0.5f),
            Random.Range(-1f, 1f)
        ).normalized;

        tempsRestant = Random.Range(tempsMinDirection, tempsMaxDirection);
    }

    void AjusterTaillePoisson()
    {
        Renderer rend = GetComponentInChildren<Renderer>();
        if (!rend) return;

        Vector3 tailleMesh = rend.bounds.size;
        if (tailleMesh == Vector3.zero) return;

        Vector3 facteur = new Vector3(
            largeurX / tailleMesh.x,
            hauteurY / tailleMesh.y,
            longueurZ / tailleMesh.z
        );

        transform.localScale = facteur;
    }

    void CalculerZoneDeNage()
    {
        if (!aquarium) return;

        BoxCollider box = aquarium.GetComponent<BoxCollider>();
        if (!box)
        {
            Debug.LogError("Aquarium doit avoir un BoxCollider.");
            return;
        }

        Bounds aquariumBounds = box.bounds;

        // Calcul du volume disponible pour la nage en soustrayant la taille du poisson
        Renderer poissonRend = GetComponentInChildren<Renderer>();
        if (!poissonRend) return;

        Vector3 marge = poissonRend.bounds.size;
        Vector3 newSize = aquariumBounds.size - marge;

        // On crée une nouvelle zone centrée comme l'aquarium
        zoneDeNage = new Bounds(aquariumBounds.center, newSize);
    }
}
