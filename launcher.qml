import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    id: root
    visible: true
    width: 640
    height: 480
    title: qsTr("Launcher")

    ColumnLayout {
        anchors.fill: parent
        Button {
            text: qsTr("Install packages")
            onClicked: installer.install_packages()
        }

        ListView {
            model: requirementsModel
            delegate: PackageDelegate {
                packageName: name
                installed: installed
            }
        }
    }

    ListModel {
        id: requirementsModel
        Component.onCompleted: {
            var requirements = []
            with (new XMLHttpRequest()) {
                open("requirements_versions.txt", "GET")
                send()
                requirements = responseText.split("\n")
            }
            for (var i = 0; i < requirements.length; i++) {
                var requirement = requirements[i]
                if (requirement.length > 0) {
                    installer.check_package(requirement)
                    requirementsModel.append({
                        "name": requirement,
                        "installed": false
                    })
                }
            }
        }
    }

    Connections {
        target: installer
        onPackageInstalled: {
            var index = -1
            for (var i = 0; i < requirementsModel.count; i++) {
                if (requirementsModel.get(i).name == package_name) {
                    index = i
                    break
                }
            }
            if (index >= 0) {
                requirementsModel.setProperty(index, "installed", installed)
            }
        }
    }
}

Item {
    id: PackageDelegate
    property string packageName
    property bool installed

    RowLayout {
        Text {
            text: packageName
        }
        Text {
            text: installed ? "Installed" : "Not installed"
            color: installed ? "green" : "red"
        }
    }
}
